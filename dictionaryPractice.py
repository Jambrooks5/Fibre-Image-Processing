dataDictionary = {}

def setOrGet(action, key, data):
    if (action=='set'):
        dataDictionary[key] = data
        print("Updated ", key, " to be ", data)
    elif (action=='get'):
        data = dataDictionary[key]
        print("Read ", key, " as ", data)
        return data
    else:
        print("Incorrect action passed to setOrGet")

#def updateData(key, data):
 #   dataDictionary[key] = data

def main():
    #global dataDictionary
    #dataDictionary['center'] = [100,200]
    setOrGet('set', 'center', [1,2])
    setOrGet('set', 'radius', 100)
    
    print(setOrGet('get', 'radius', 0))
    
    
    print(dataDictionary)
    
    setOrGet('set', 'radius', 200)
    
    print(dataDictionary)
    
main()