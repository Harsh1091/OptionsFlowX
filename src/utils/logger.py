"""
Logging utilities for OptionsFlowX.

This module provides centralized logging configuration using loguru
for consistent logging across the application.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import get_config


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Custom log format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format if not provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=max_file_size,
            retention=backup_count,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str = "OptionsFlowX"):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


def setup_logger_from_config() -> None:
    """
    Setup logger using configuration from settings file.
    """
    try:
        config = get_config()
        logging_config = config.get('logging', {})
        
        log_level = logging_config.get('level', 'INFO')
        log_file = None
        
        if logging_config.get('file_enabled', True):
            log_file = "data/logs/optionsflowx.log"
        
        max_file_size = logging_config.get('max_file_size', '10MB')
        backup_count = logging_config.get('backup_count', 5)
        format_string = logging_config.get('format')
        
        setup_logger(
            log_level=log_level,
            log_file=log_file,
            max_file_size=max_file_size,
            backup_count=backup_count,
            format_string=format_string
        )
        
    except Exception as e:
        # Fallback to basic logging if config fails
        setup_logger()
        logger.warning(f"Failed to setup logger from config: {e}")


def log_performance_metrics(metrics: dict) -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        metrics: Dictionary containing performance metrics
    """
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


def log_trading_signal(signal: dict) -> None:
    """
    Log trading signal information.
    
    Args:
        signal: Dictionary containing signal information
    """
    logger.info("Trading Signal Generated:")
    logger.info(f"  Symbol: {signal.get('symbol', 'N/A')}")
    logger.info(f"  Signal Type: {signal.get('signal_type', 'N/A')}")
    logger.info(f"  Strength: {signal.get('strength', 'N/A')}")
    logger.info(f"  Price: {signal.get('price', 'N/A')}")
    logger.info(f"  Timestamp: {signal.get('timestamp', 'N/A')}")


def log_position_update(position: dict) -> None:
    """
    Log position update information.
    
    Args:
        position: Dictionary containing position information
    """
    logger.info("Position Update:")
    logger.info(f"  Symbol: {position.get('symbol', 'N/A')}")
    logger.info(f"  Action: {position.get('action', 'N/A')}")
    logger.info(f"  Quantity: {position.get('quantity', 'N/A')}")
    logger.info(f"  Price: {position.get('price', 'N/A')}")
    logger.info(f"  P&L: {position.get('pnl', 'N/A')}")


def log_error_with_context(error: Exception, context: dict = None) -> None:
    """
    Log error with additional context information.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger.error(f"Error occurred: {str(error)}")
    if context:
        logger.error(f"Context: {context}")
    logger.exception("Full traceback:")


def log_market_data_update(symbol: str, data: dict) -> None:
    """
    Log market data update (debug level).
    
    Args:
        symbol: Trading symbol
        data: Market data dictionary
    """
    logger.debug(f"Market data update for {symbol}: {data}")


def log_indicator_calculation(indicator: str, symbol: str, value: float) -> None:
    """
    Log indicator calculation result (debug level).
    
    Args:
        indicator: Indicator name
        symbol: Trading symbol
        value: Calculated value
    """
    logger.debug(f"{indicator} for {symbol}: {value}")


# Initialize logger when module is imported
if not logger._core.handlers:
    setup_logger_from_config() 