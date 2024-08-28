import numpy as np

# Function
def greet(name):
    """
    This function greets the person passed in as an argument.
    name: str: input name
    """
    print("My name is {}".format(name))
    print(f"My name is {name}")
    return name

name = greet("Dung")


# Class
class Car:
    def __init__(self, make, model, year) -> None:
        self.make = make
        self.model = model
        self.year = year
        
    
    def update_info(self, year):
        self.year = year


car_1 = Car("Toyota", "Cambr", 1995)
print(car_1.make, car_1.model, car_1.year)
car_1.update_info(year=2000)
print(car_1.make, car_1.model, car_1.year)


# Class
class Model:
    def __init__(self, x=0, y=0, z=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.init_weights = np.array([1, 2, 3])
        self.result = None
        
    
    def plus(self, year):
        self.result = self.x + self.y + self.z
        return self.result
    
    def optimize(self):
        self.init_weights += np.array([1, 1, 1])

model = Model()
for i in range(5):
    model.optimize()
    print(model.init_weights)


# Superclass (Parent)
class Animal:
    def __init__(self, species, sound) -> None:
        self.species = species
        self.sound = sound

    def make_sound(self):
        print(f"The {self.species} makes a sound: {self.sound}")


# subclass
class Dog(Animal):
    def __init__(self, name, breed) -> None:
        super().__init__("Dog", "Bark")
        self.name = name
        self.breed = breed

    def describe_dog(self):
        print(self.name, self.breed)


class Cat(Animal):
    def __init__(self, name, color) -> None:
        super().__init__("Cat", "Meow")
        self.name = name
        self.color = color

    def describe_cat(self):
        print(self.name, self.color)

# Creating instances of the subclasses
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Red")

dog.describe_dog()
cat.describe_cat()

dog.make_sound()
cat.make_sound()


    



    