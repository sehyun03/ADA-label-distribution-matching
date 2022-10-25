class AverageMeter(object):
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict, denominator=None):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            if denominator is None:
                self.__data[k][1] += 1
            else:
                self.__data[k][1] += denominator

    def get(self, *keys):
        if len(keys) == 1:
            try:
                return self.__data[keys[0]][0] / self.__data[keys[0]][1]
            except:
                return 0
        else:
            v_list = [self.get(k) for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
    
    def get_whole_data(self):
        return self.__data