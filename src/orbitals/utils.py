import logging
import os


def maybe_configure_gpu_usage():
    """
    If the user hasn't explicitly set CUDA_VISIBLE_DEVICES, auto-configure it for
    optimal usage: search for the gpu with the most free memory, and
    set CUDA_VISIBLE_DEVICES to that GPU only.

    Uses nvidia-ml-py (pynvml) to avoid triggering CUDA initialization from torch.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info(
            "CUDA_VISIBLE_DEVICES is already set, not auto-configuring GPU usage"
        )
        return

    try:
        import pynvml  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        logging.debug("nvidia-ml-py not available, skipping GPU auto-configuration")
        return

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logging.error("NVML initialization failed, skipping GPU auto-configuration")
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return

        best_idx = None
        best_free = -1

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem_info.free > best_free:
                best_idx = i
                best_free = mem_info.free

        if best_idx is not None:
            if device_count > 1:
                logging.warning(
                    f"Auto-configuring to use GPU {best_idx} with {best_free / 1024**3:.2f} GB free"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    finally:
        pynvml.nvmlShutdown()
