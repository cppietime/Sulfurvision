import pyopencl as cl

def create_ctx() -> cl.Context:
    return cl.create_some_context(interactive=False)

def pick_device(ctx: cl.Context) -> cl.Device:
    # Currently just returns the first GPU if there is one, else the first device
    devices = ctx.devices
    gpus = list(filter(lambda d: d.type == cl.device_type.GPU, devices))
    if not gpus:
        return devices[0]
    return gpus[0]
