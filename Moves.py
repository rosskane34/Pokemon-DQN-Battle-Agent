class Move:
    def __init__(self,name,type,power,category,effect):
        self.name=name
        self.type=type              #store the moves name, type, power, category and effect
        self.power=power
        self.category=category
        self.effect=effect
    def getName(self):          #getter functions
        return self.name
    def getType(self):
        return self.type
    def getPower(self):
        return self.power
    def getCategory(self):
        return self.category
    def getEffect(self):
        return self.effect
MOVES= {
   "Tackle" :Move("Tackle","Normal",40,0,0),       #We use 0 for no effect, 1 for attackDrop Effect Moves
   "Growl"  :Move("Growl","Normal",0,2,1),           #We use 0 for Physical Moves and 1 for Special Moves and 2 for status
   "Bite"   :Move("Bite","Dark",30,1,0),            #This is a global dictionary of all moves implemented in our system currently
   "Flamethrower": Move("Flamethrower","Fire",80,1,0),
   "Waterfall" : Move("Waterfall","Water",80,1,0),
   "Vine Whip": Move("Vine Whip","Grass",80,1,0)
}
