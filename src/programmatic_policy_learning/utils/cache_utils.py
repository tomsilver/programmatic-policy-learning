"""Cache utilities."""

import os
import pickle
from typing import Any, Callable, Iterable, Union

from scipy.sparse import csr_matrix, load_npz, save_npz


def cache_single_output(output: Any, cache_file: str) -> None:
    """Persist a single cached output to disk.

    If the output is a SciPy CSR matrix it is written with ``save_npz`` and
    otherwise pickled. The function prints a short confirmation message.

    Args:
        output: The Python object to cache.
        cache_file: The path to write the cached object to.
    """
    if isinstance(output, csr_matrix):
        save_npz(cache_file, output)
    else:
        with open(cache_file, "wb") as f:
            pickle.dump(output, f)
    print(f"Cached output to {cache_file}.")


def load_single_cache_output(cache_file: str) -> Any:
    """Load a single cached output from disk.

    Args:
        cache_file: Path to the cache file produced by :func:`cache_single_output`.

    Returns:
        The Python object that was stored in the cache file.
    """
    if ".npz" in cache_file:
        output = load_npz(cache_file)
    else:
        with open(cache_file, "rb") as f:
            output = pickle.load(f)

    print(f"Loaded cache from {cache_file}.")
    return output


def manage_cache(
    cache_dir: str, extensions: Union[str, Iterable[str]]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that caches function outputs under ``cache_dir``.

    Cached files are named ``<func>_<run_id>_<i>.<ext>``. If ``extensions`` is
    a string the function should return a single value; otherwise it should
    return a tuple matching the provided extensions.

    Args:
        cache_dir: Directory where cache files will be written.
        extensions: Either a single extension (e.g. '.pkl', '.npz') or an
            iterable of extensions if the function returns multiple outputs.

    Returns:
        A decorator that can be applied to functions.
    """

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if isinstance(extensions, str):
        single_output = True
        extensions_list: list[str] = [extensions]
    else:
        single_output = False
        extensions_list = list(extensions)

    def decorator_manage_cache(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper_cache_output(*args: Any, **kwargs: Any) -> Any:
            # Create a simple run id from positional args
            run_id = "-".join([str(arg) for arg in args])

            # Check the existence of the first cache file for this run
            cache_file = os.path.join(
                cache_dir, f"{func.__name__}_{run_id}_{0}{extensions_list[0]}"
            )

            if not os.path.isfile(cache_file):
                func_outputs = func(*args, **kwargs)
                if single_output:
                    func_outputs = [func_outputs]

                for i, (output, extension) in enumerate(
                    zip(func_outputs, extensions_list)
                ):
                    cache_file = os.path.join(
                        cache_dir, f"{func.__name__}_{run_id}_{i}{extension}"
                    )
                    cache_single_output(output, cache_file)

            # Load cached files and return them in the same shape as the
            # original function's return value.
            outputs: list[Any] = []
            for i, extension in enumerate(extensions_list):
                cache_file = os.path.join(
                    cache_dir, f"{func.__name__}_{run_id}_{i}{extension}"
                )
                output = load_single_cache_output(cache_file)
                outputs.append(output)

            if single_output:
                return outputs[0]
            return tuple(outputs)

        return wrapper_cache_output

    return decorator_manage_cache
