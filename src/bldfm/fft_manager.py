import pyfftw
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import pyfftw.interfaces.cache
import pickle
import atexit
from pathlib import Path
from .utils import get_logger

logger = get_logger(__name__.split("bldfm.")[-1])


class FFTManager:
    """
    Manages pyFFTW with wisdom and caching for optimal performance in Dask environments.

    This class addresses memory issues in Dask parallelized environments by:
    - Using pyfftw numpy interface for compatibility
    - Loading/saving FFTW wisdom for optimal planning
    - Enabling pyfftw caching to prevent repeated object allocation
    - Providing proper memory management and cleanup
    """

    def __init__(
        self, wisdom_file="fftw_wisdom.pkl", num_threads=1, cache_keepalive=30
    ):
        self.wisdom_file = Path(wisdom_file)
        self.num_threads = num_threads

        # Configure threading
        pyfftw.config.NUM_THREADS = num_threads

        # Enable caching to prevent memory issues in Dask workers
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(cache_keepalive)

        # Load existing wisdom
        self._load_wisdom()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        logger.info(
            f"FFTManager initialized with {num_threads} threads, cache keepalive {cache_keepalive}s"
        )

    def _load_wisdom(self):
        """Load FFTW wisdom from file if it exists."""
        try:
            if self.wisdom_file.exists():
                with open(self.wisdom_file, "rb") as f:
                    wisdom = pickle.load(f)
                    import_results = pyfftw.import_wisdom(wisdom)
                    if all(import_results):
                        logger.info(
                            f"Successfully loaded wisdom from {self.wisdom_file}"
                        )
                    else:
                        logger.warning(
                            "Wisdom import partially failed, will be regenerated"
                        )
            else:
                logger.info(
                    "No existing wisdom file found, will be generated during first use"
                )
        except Exception as e:
            logger.warning(f"Failed to load wisdom: {e}")

    def _save_wisdom(self):
        """Save FFTW wisdom to file."""
        try:
            wisdom = pyfftw.export_wisdom()
            with open(self.wisdom_file, "wb") as f:
                pickle.dump(wisdom, f)
            logger.debug(f"Wisdom saved to {self.wisdom_file}")
        except Exception as e:
            logger.warning(f"Failed to save wisdom: {e}")

    def fft2(self, input_data, norm="backward"):
        """
        Perform 2D forward FFT using pyfftw numpy interface.

        Parameters:
            input_data: Input array for FFT
            norm: Normalization mode ("forward", "backward", "ortho")

        Returns:
            Complex array with FFT result
        """
        return pyfftw_fft.fft2(input_data, norm=norm)

    def ifft2(self, input_data, norm="backward"):
        """
        Perform 2D inverse FFT using pyfftw numpy interface.

        Parameters:
            input_data: Input array for inverse FFT
            norm: Normalization mode ("forward", "backward", "ortho")

        Returns:
            Complex array with inverse FFT result
        """
        return pyfftw_fft.ifft2(input_data, norm=norm)

    def clear_cache(self):
        """Clear pyfftw cache."""
        pyfftw.interfaces.cache.disable()
        pyfftw.interfaces.cache.enable()
        logger.debug("PyFFTW cache cleared")

    def _cleanup(self):
        """Cleanup function called on exit."""
        self._save_wisdom()
        self.clear_cache()
        logger.debug("FFTManager cleanup completed")


# Global FFT manager instance
_fft_manager = None


def get_fft_manager(num_threads=1, cache_keepalive=30):
    """Get or create the global FFT manager instance.

    If the manager already exists with different num_threads, it is
    re-initialised with the new thread count.
    """
    global _fft_manager
    if _fft_manager is not None and _fft_manager.num_threads != num_threads:
        logger.info(
            "Re-initializing FFTManager: num_threads %d -> %d",
            _fft_manager.num_threads, num_threads,
        )
        _fft_manager = None
    if _fft_manager is None:
        _fft_manager = FFTManager(
            num_threads=num_threads, cache_keepalive=cache_keepalive
        )
    return _fft_manager


def reset_fft_manager():
    """Reset FFTManager singleton (call in forked worker processes)."""
    global _fft_manager
    _fft_manager = None


def fft2(input_data, norm="backward"):
    """Global function for 2D FFT using the FFT manager."""
    manager = get_fft_manager()
    return manager.fft2(input_data, norm)


def ifft2(input_data, norm="backward"):
    """Global function for 2D inverse FFT using the FFT manager."""
    manager = get_fft_manager()
    return manager.ifft2(input_data, norm)
