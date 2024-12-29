# Usually all this values or most of them would be read from Config files via os lib
class Config:
    # Usually not available public, but fetched from secrets via reference, example AWS secrets
    AUTHORIZATION_KEY = "testingAPIKEY*"
    AUTH_FAILED_STATUS_CODE = 401
    HEADER_AUTH_ALIAS = "Authorization"
    APP_NAME = "ranking-search-engine-api"
    BASE_API_PATH = "/ranking-engine"
    LOGGER_NAME = APP_NAME
    NO_OF_RESULTS_RETURNED = 10
    UPDATE_TD_MATRIX = False
    USE_REDIS = False
    REDIS_TTL = 172800  # in seconds Time to live 48 h
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_USERNAME = "user"
    REDIS_PASSWORD = "password"
    REDIS_SSL = True
