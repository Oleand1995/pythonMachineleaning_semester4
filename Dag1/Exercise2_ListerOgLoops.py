
forfattere = []
forfattere.append("J.k.R")
forfattere.append("Frank Petersen")
forfattere.append("Jens jensen")
forfattere.append("Ole person")
forfattere.append("Snaller Snallesen")

def LoopThrowList():
    for forfatter in forfattere:
        print(forfatter)

def deleteFromList(index = 0):
    forfattere.pop(index)

def addToList(navn):
    forfattere.append(navn)

def LengthOnTheList(list):
    size = len(list)
    return size


LoopThrowList()
print()

deleteFromList(2)
LoopThrowList()
print()

addToList("Per ole larsen")
LoopThrowList()
print()

deleteFromList()
LoopThrowList()
print()

print(LengthOnTheList(forfattere))
print()

forfattere.reverse()
LoopThrowList()
print()


