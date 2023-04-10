class Person:
    def Person(self, name):
        self.name = name
        self.cars = []

    def add_car(self, car):
        self.cars.append(car)
        print(f"{car} has been added to {self.name}'s cars.");

