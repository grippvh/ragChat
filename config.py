class Config:
    # JINA_API_KEY = os.getenv("JINA_API_KEY", "your_default_key")
    # COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your_default_key")
    JINA_API_KEY = "jina_ba01f74af583437c80a14daa9ddf43f2FHVkdLV9s9gUWEsBYmicVIDahLX9"  # from https://jina.ai/embeddings
    COHERE_API_KEY = "7oeDYIEeWooiPs1hLmJ9mwNKIn7SvwJSXQxoOSF9"
    DB_PATH = "db"
    TEST_DB_PATH = "test_db"
    DATA_PATH = "data"
    UNUSED_DATA_PATH = "unused_data"
    MODEL_NAME = "llama3"
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/56.0.2924.76 Safari/537.36',
        "Upgrade-Insecure-Requests": "1", "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate"
    }