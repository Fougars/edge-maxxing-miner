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
HF_REPOSITORY = "Lucas67/edge-maxxing-miner"  # Updated repository name
GITHUB_REPOSITORY = "https://github.com/fougars/bittensor-subnet39-miner"
MODEL_ID = "stablediffusionapi/newdream-sdxl-20"

class CrossAttentionPlugin(trt.IPluginV2):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def get_output_shape(self, input_shape):
        batch_size, seq_length, _ = input_shape
        return (batch_size, seq_length, self.hidden_size)

    def configure_plugin(self, in_out):
        in_out[0].dtype = trt.DataType.HALF
        in_out[0].shape = self.get_output_shape(in_out[0].shape)
        in_out[1].dtype = trt.DataType.HALF
        in_out[1].shape = self.get_output_shape(in_out[1].shape)
        return in_out

    def execute_plugin(self, inputs, outputs, stream, batch_size):
        query, key, value = inputs
        attention_scores = trt.matmul(query, key.transpose(1, 2))
        attention_scores = trt.softmax(attention_scores, dim=-1)
        attention_output = trt.matmul(attention_scores, value)
        outputs[0] = attention_output
        return 0

def replace_cross_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hidden_size = module.embed_dim
            num_heads = module.num_heads
            plugin = CrossAttentionPlugin(hidden_size, num_heads)
            layer = trt.CustomLayer(plugin, name)
            model.replace_module(name, lambda *inputs: layer(*inputs))
    return model

def build_engine(onnx_file_path, precision='fp16', max_batch_size=256):
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
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 4, 128, 128), (max_batch_size // 2, 4, 128, 128), (max_batch_size, 4, 128, 128))
    config.add_optimization_profile(profile)

    if builder.num_DLA_cores:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = 0
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    return builder.build_engine(network, config)

def allocate_buffers(engine, max_batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        shape = (max_batch_size,) + shape[1:]
        size = trt.volume(shape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

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
    model = replace_cross_attention(model)
    model = model.eval().half().cuda()

    from torch.nn.utils import prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    dummy_input = torch.randn(*input_shape).half().cuda() 

    torch.onnx.export(model, dummy_input, f"{model.__class__.__name__}.onnx", opset_version=13,
                      do_constant_folding=True, input_names=['input'],  
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 
                                                             'output': {0: 'batch_size'}})

    engine = build_engine(f"{model.__class__.__name__}.onnx", precision, max_batch_size)
    inputs, outputs, bindings, stream = allocate_buffers(engine, max_batch_size)
    context = engine.create_execution_context()

    def inference_fn(input_data):
        batch_size = get_adaptive_batch_size(max_batch_size)
        input_data = input_data[:batch_size]
        context.set_binding_shape(0, (batch_size,) + input_data.shape[1:])
        cuda.memcpy_htod_async(inputs[0].device, input_data.ravel(), stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream) 
        stream.synchronize()
        return outputs[0].host.reshape((batch_size,) + engine.get_binding_shape(1)[1:])

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
