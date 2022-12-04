# whisper-py

Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

_Latest working commit on the main branch: **4698dcd**_

## How to setup

1. Clone this repository.
2. Clone [whisper.cpp](https://github.com/ggerganov/whisper.cpp).
3. Copy `pybind.sh` and `whisper.py` from this repository to the root folder of `whisper.cpp`.
4. Download a base language model according to [whisper.cpp](https://github.com/ggerganov/whisper.cpp)'s `README`
5. Run: `sh pybind.sh`.
6. Test if everything works with: `python whisper.py`.
7. You can build on `whisper.py` further.

## License

See the [LICENSE](./LICENSE) file for more details.
