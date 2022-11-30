import ctypes
import pathlib

from scipy.io import wavfile


libname     = "libwhisper.so"
fname_model = "models/ggml-base.en.bin"
fname_wav   = "samples/jfk.wav"
is_verbose = True

# this needs to match the C struct in ./whisper.h
class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy",             ctypes.c_int),
        ("n_threads",            ctypes.c_int),
        ("n_max_text_ctx",       ctypes.c_int),
        ("offset_ms",            ctypes.c_int),
        ("duration_ms",          ctypes.c_int),

        ("translate",            ctypes.c_bool),
        ("no_context",           ctypes.c_bool),
        ("single_segment",       ctypes.c_bool),
        ("print_special",        ctypes.c_bool),
        ("print_progress",       ctypes.c_bool),
        ("print_realtime",       ctypes.c_bool),
        ("print_timestamps",     ctypes.c_bool),

        ("token_timestamps",     ctypes.c_bool),
        ("thold_pt",             ctypes.c_float),
        ("thold_ptsum",          ctypes.c_float),
        ("max_len",              ctypes.c_int),
        ("max_tokens",           ctypes.c_int),
        
        ("speed_up",             ctypes.c_bool),
        ("audio_ctx",            ctypes.c_int),

        ("prompt_tokens",        ctypes.c_int * 16),
        ("prompt_n_tokens",      ctypes.c_int),
        
        ("language",             ctypes.c_char_p),
        ("greedy",               ctypes.c_int * 1),
        ("beam_search",          ctypes.c_int * 3),
    ]


def format_time(t: int) -> str:
    millis = int(t*10)
    seconds = (millis/1000)%60
    seconds = int(seconds)
    minutes = (millis/(1000*60))%60
    minutes = int(minutes)
    hours = (millis/(1000*60*60))%24
    hours = int(hours)
    millis -= 1000*seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


if __name__ == "__main__":
    libname = pathlib.Path().absolute() / libname
    whisper = ctypes.CDLL(libname)

    if is_verbose:
        print("tell Python what are the return types of the functions")
    whisper.whisper_init.restype                  = ctypes.c_void_p
    whisper.whisper_full_default_params.restype   = WhisperFullParams
    whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p

    if is_verbose:
        print("initialize whisper.cpp context")
    ctx = whisper.whisper_init(fname_model.encode("utf-8"))

    if is_verbose:
        print("get default whisper parameters and adjust as needed")
    params = whisper.whisper_full_default_params(0)

    if is_verbose:
        print("load WAV file")
    samplerate, data = wavfile.read(fname_wav)

    if is_verbose:
        print("convert to 32-bit float")
    data = data.astype('float32')/32768.0

    if is_verbose:
        print("run the inference")
    result = whisper.whisper_full(
        ctypes.c_void_p(ctx),
        params,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(data)
    )

    if result != 0:
        print("Error: {}".format(result))
        exit(1)

    if is_verbose:
        print("\nResults from Python:\n")
    
    n_segments = whisper.whisper_full_n_segments(ctypes.c_void_p(ctx))
    for i in range(n_segments):
        t0  = whisper.whisper_full_get_segment_t0(ctypes.c_void_p(ctx), i)
        t1  = whisper.whisper_full_get_segment_t1(ctypes.c_void_p(ctx), i)
        txt = whisper.whisper_full_get_segment_text(ctypes.c_void_p(ctx), i)

        print(f"[{format_time(t0)} - {format_time(t1)}]: {txt.decode('utf-8')}")

    if is_verbose:
        print("free the memory")
        
    whisper.whisper_free(ctypes.c_void_p(ctx))
