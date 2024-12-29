import logging

from fastapi import Header, HTTPException
import redis as redis
from app.config import Config


def verify_key(secret_key: str = Header(
    default=...,
    alias=Config.HEADER_AUTH_ALIAS,
    description="Authorization secret key")
):
    """
    Function to provide basic security
    """
    if secret_key != Config.AUTHORIZATION_KEY:
        raise HTTPException(
            status_code=Config.AUTH_FAILED_STATUS_CODE,
            detail="Authorization secret key invalid"
        )


def connect_redis(
        host,
        port,
        password=None,
        username=None,
        db=0,
        socket_timeout=5,
        ssl=True
) -> redis.client.Redis:
    client = redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        socket_timeout=socket_timeout,
        ssl=ssl,
        username=username
    )
    try:
        client.ping()
        logging.info("Redis client connected")
    except Exception as e:
        logging.exception(f"Redis client unable to connect, error: {repr(e)}")
        return None
    return client
