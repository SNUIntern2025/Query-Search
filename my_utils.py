from datetime import datetime

# 함수 소요시간 측정 decorator
def timeit(method):
    def timed(*args, **kw):
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        print(f"{method.__name__} 소요시간: {te-ts}")
        return result
    return timed