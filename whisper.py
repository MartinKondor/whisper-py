import ctypes
import pathlib

from scipy.io import wavfile


whisper_folder = "."
libname     = f"{whisper_folder}/libwhisper.so"
fname_model = f"{whisper_folder}/models/ggml-base.en.bin"
fname_wav   = f"{whisper_folder}/samples/jfk.wav"
language = b"en"
is_verbose = False

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


def whisper_full_default_params(def_params: WhisperFullParams) -> WhisperFullParams:
    params = WhisperFullParams()
    params.strategy = 0
    params.n_threads = 1
    params.n_max_text_ctx = 16384
    params.offset_ms = 0
    params.duration_ms = 0
    params.translate = 0
    params.no_context = 0
    params.single_segment = 0
    params.print_special = 0
    params.print_progress = 1
    params.print_realtime = 0
    params.print_timestamps = 1
    params.token_timestamps = 0
    params.thold_pt = 0.01
    params.thold_ptsum = 0.01
    params.max_len = 0
    params.max_tokens = 0
    params.speed_up = 0
    params.audio_ctx = 0
    params.prompt_tokens = def_params.prompt_tokens
    params.prompt_n_tokens = 0
    params.language = def_params.language
    params.greedy = def_params.greedy
    params.beam_search = def_params.beam_search
    params.new_segment_callback = None
    params.new_segment_callback_user_data = None
    params.encoder_begin_callback = None
    params.encoder_begin_callback_user_data = None
    return params


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
    whisper._whisper_full_default_params.restype   = WhisperFullParams
    whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p

    if is_verbose:
        print("initialize whisper.cpp context")
    ctx = whisper.whisper_init(fname_model.encode("utf-8"))

    if is_verbose:
        print("get default whisper parameters and adjust as needed")
    _params = whisper._whisper_full_default_params(0, language)
    params = whisper_full_default_params(_params)

    # set parameters
    params.print_special = is_verbose
    params.print_progress = is_verbose
    params.print_realtime = is_verbose
    params.print_timestamps = is_verbose

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
    