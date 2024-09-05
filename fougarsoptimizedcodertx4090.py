import bittensor as bt
from diffusers import DiffusionPipeline
from huggingface_hub import upload_folder
from neuron import (
    CheckpointSubmission,
    get_submission,
    get_config,
    compare_checkpoints,
    CURRENT_CONTEST,
    make_submission,
)
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time
import psutil

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MODEL_DIRECTORY = "model"
MODEL_ID = "stablediffusionapi/newdream-sdxl-20"

class TensorRTEngine:
    def __init__(self, onnx_file_path, precision='fp16', max_batch_size=256):
        self.engine = self.build_engine(onnx_file_path, precision, max_batch_size)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(max_batch_size)

    def build_engine(self, onnx_file_path, precision='fp16', max_batch_size=256):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config.max_workspace_size = 24 * (1 << 30)  # 24 GB
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 4, 128, 128), (max_batch_size // 2, 4, 128, 128), (max_batch_size, 4, 128, 128))
        config.add_optimization_profile(profile)

        return builder.build_engine(network, config)

    def allocate_buffers(self, max_batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            shape = (max_batch_size,) + shape[1:]
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        batch_size = input_data.shape[0]
        self.context.set_binding_shape(0, (batch_size,) + input_data.shape[1:])
        cuda.memcpy_htod_async(self.inputs[0].device, input_data.ravel(), self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()
        return self.outputs[0].host.reshape((batch_size,) + self.engine.get_binding_shape(1)[1:])

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def get_adaptive_batch_size(max_batch_size=256):
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = pycuda.driver.Device(0).get_attribute(pycuda.driver.device_attribute.GPU_UTIL_RATE)
    
    if cpu_usage > 90 or memory_usage > 90 or gpu_usage > 90:
        return max(1, max_batch_size // 4)
    elif cpu_usage > 70 or memory_usage > 70 or gpu_usage > 70:
        return max(1, max_batch_size // 2)
    else:
        return max_batch_size

def optimize_model(model, input_shape, precision='fp16', max_batch_size=256):
    model = model.eval().half().cuda()

    # Pruning
    from torch.nn.utils import prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    # Quantization
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Export to ONNX
    dummy_input = torch.randn(*input_shape).half().cuda() 
    onnx_file = f"{model.__class__.__name__}.onnx"
    torch.onnx.export(model, dummy_input, onnx_file, opset_version=13,
                      do_constant_folding=True, input_names=['input'],  
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 
                                                             'output': {0: 'batch_size'}})

    # Create TensorRT engine
    trt_engine = TensorRTEngine(onnx_file, precision, max_batch_size)

    def inference_fn(input_data):
        batch_size = get_adaptive_batch_size(max_batch_size)
        input_data = input_data[:batch_size]
        return trt_engine.infer(input_data)

    return inference_fn

def optimize(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    max_batch_size = 256  
    precision = 'fp16'  

    # Optimize for RTX 4090
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    with ThreadPoolExecutor(max_workers=4) as executor:
        unet_future = executor.submit(optimize_model, pipeline.unet, (2, 4, 128, 128), precision, max_batch_size)
        text_encoder_future = executor.submit(optimize_model, pipeline.text_encoder, (1, 77), precision, max_batch_size)
        vae_decoder_future = executor.submit(optimize_model, pipeline.vae.decoder, (1, 4, 128, 128), precision, max_batch_size)
        vae_encoder_future = executor.submit(optimize_model, pipeline.vae.encoder, (1, 3, 1024, 1024), precision, max_batch_size)

        optimized_unet = unet_future.result()
        optimized_text_encoder = text_encoder_future.result()
        optimized_vae_decoder = vae_decoder_future.result()
        optimized_vae_encoder = vae_encoder_future.result()

    pipeline.unet = lambda x, t, c: optimized_unet(np.concatenate([x, c], axis=1))
    pipeline.text_encoder = lambda x: optimized_text_encoder(x)
    pipeline.vae.decoder = lambda x: optimized_vae_decoder(x)
    pipeline.vae.encoder = lambda x: optimized_vae_encoder(x)

    return pipeline

def main():
    config = get_config()
    config.wallet.name = "fougarsHK"
    config.wallet.hotkey = "5HWJnmizA4cwbf7AHguxDMnssgc6HjxRwff9NskLfvK4GEhy"
    config.repository = "https://huggingface.co/Lucas67/edge-maxxing-miner"

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    wallet = bt.wallet(config=config)

    CURRENT_CONTEST.validate()

    if os.path.isdir(MODEL_DIRECTORY):
        repository = MODEL_DIRECTORY
    else:
        repository = MODEL_ID

    start_time = time.time()
    pipeline = DiffusionPipeline.from_pretrained(repository, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline = optimize(pipeline)
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.2f} seconds")

    pipeline.save_pretrained(MODEL_DIRECTORY)

    repository = MODEL_DIRECTORY

    comparison = compare_checkpoints(CURRENT_CONTEST, repository)

    if config.commit:
        if comparison.failed:
            bt.logging.warning("Not pushing to huggingface as the checkpoint failed to beat the baseline.")
            return

        upload_folder(repo_id=config.repository, folder_path=MODEL_DIRECTORY, commit_message="Updated optimized SDXL model")
        bt.logging.info(f"Pushed to huggingface at {config.repository}")

    checkpoint_info = CheckpointSubmission(
        repository=config.repository,
        average_time=comparison.average_time,
    )

    make_submission(
        subtensor,
        metagraph,
        wallet,
        checkpoint_info,
    )

    bt.logging.info(f"Submitted {checkpoint_info} as the info for this miner")

if __name__ == '__main__':
    main()
