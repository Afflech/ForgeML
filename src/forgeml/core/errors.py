class ForgeError(Exception):
    """Base error for all ForgeML failures."""


class ConfigError(ForgeError):
    """Invalid or missing configuration."""


class CapabilityError(ForgeError):
    """Requested model/dataset/category not supported."""


class ProviderError(ForgeError):
    """Kaggle API or network failure."""


class QuotaError(ProviderError):
    """Kaggle GPU quota exhausted."""


class AuthError(ProviderError):
    """Kaggle authentication failed."""


class PackagingError(ForgeError):
    """Failed to create source bundle."""


class ArtifactError(ForgeError):
    """Failed to download or verify artifacts."""


class LockError(ForgeError):
    """Another run is already active for this project."""
