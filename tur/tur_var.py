# Global variables
global_var = "global var"

# loval variables
def exp_func(input_params=None):
    local_var = "local var"
    return local_var

# Lists
fruits = ["apple", "orange"]
fruits.append("peach")
print(fruits)

# Tuples
color = ('red', 'green', 'nlue')

# Dictionaries --> json
person_1 = {
    'id': 12321341234,
    'name': "Huy",
    'age': 21,
    'email': ['@gmail', '@hotmail']
}

person_2 = {
    'id': 12321341234,
    'name': "Huy",
    'age': 21,
    'email': ['@gmail', '@hotmail']
}

persons = {
    'persons': [person_1, person_2]
}

for k, v in person_1.items():
    print(k, v)

