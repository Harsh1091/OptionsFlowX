"""
Configuration management utilities for OptionsFlowX.

This module provides functions to load, validate, and manage configuration
settings from YAML files and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

# Global configuration cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Override with environment variables
        config = _override_with_env_vars(config)
        
        # Validate configuration
        _validate_config(config)
        
        _config_cache = config
        logger.info(f"Configuration loaded successfully from {config_path}")
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str = "config/settings.yaml") -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved successfully to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Current configuration dictionary
    """
    if _config_cache is None:
        return load_config()
    return _config_cache


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Updated configuration with environment variable overrides
    """
    env_mappings = {
        'OPTIONSFLOWX_API_KEY': ('api', 'api_key'),
        'OPTIONSFLOWX_API_SECRET': ('api', 'api_secret'),
        'OPTIONSFLOWX_ACCESS_TOKEN': ('api', 'access_token'),
        'OPTIONSFLOWX_PROVIDER': ('api', 'provider'),
        'OPTIONSFLOWX_LOG_LEVEL': ('logging', 'level'),
        'OPTIONSFLOWX_DB_PATH': ('database', 'path'),
    }
    
    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            _set_nested_value(config, config_path, env_value)
            logger.debug(f"Overriding {config_path} with environment variable {env_var}")
    
    return config


def _set_nested_value(config: Dict[str, Any], path: tuple, value: Any) -> None:
    """
    Set a nested value in configuration dictionary.
    
    Args:
        config: Configuration dictionary
        path: Tuple of keys representing the path
        value: Value to set
    """
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['api', 'trading', 'indicators', 'signal_processing']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate API configuration
    api_config = config.get('api', {})
    if api_config.get('provider') not in ['zerodha', 'angel', 'upstox', 'paper_trading']:
        raise ValueError("Invalid API provider specified")
    
    # Validate trading parameters
    trading_config = config.get('trading', {})
    if trading_config.get('max_positions', 0) <= 0:
        raise ValueError("max_positions must be greater than 0")
    
    if trading_config.get('stop_loss_percent', 0) <= 0:
        raise ValueError("stop_loss_percent must be greater than 0")
    
    # Validate indicator parameters
    indicators_config = config.get('indicators', {})
    rsi_config = indicators_config.get('rsi', {})
    if rsi_config.get('period', 0) <= 0:
        raise ValueError("RSI period must be greater than 0")
    
    ema_config = indicators_config.get('ema', {})
    if ema_config.get('short_period', 0) >= ema_config.get('long_period', 0):
        raise ValueError("EMA short period must be less than long period")
    
    logger.info("Configuration validation completed successfully")


def reload_config() -> Dict[str, Any]:
    """
    Reload configuration from file, clearing cache.
    
    Returns:
        Fresh configuration dictionary
    """
    global _config_cache
    _config_cache = None
    return load_config()


def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration section.
    
    Returns:
        API configuration dictionary
    """
    config = get_config()
    return config.get('api', {})


def get_trading_config() -> Dict[str, Any]:
    """
    Get trading configuration section.
    
    Returns:
        Trading configuration dictionary
    """
    config = get_config()
    return config.get('trading', {})


def get_indicators_config() -> Dict[str, Any]:
    """
    Get indicators configuration section.
    
    Returns:
        Indicators configuration dictionary
    """
    config = get_config()
    return config.get('indicators', {})


def get_signal_processing_config() -> Dict[str, Any]:
    """
    Get signal processing configuration section.
    
    Returns:
        Signal processing configuration dictionary
    """
    config = get_config()
    return config.get('signal_processing', {}) 