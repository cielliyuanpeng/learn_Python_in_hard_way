# create a mapping of state to abbreviation
states = {
    'Oregon':'OR',
    'Florida':'FL',
    'California':'CA',
    'New York':'NY',
    'Michigan':'MI'
}

cities = {
    'CA':'San Francisco',
    'MI':'Detroit',
    'FL':'Jacksoville'
}

cities['NY'] = 'New York'
cities['OR'] = 'Portland'

#print cities
print('-'*10)
print('NY state has',cities['NY'])
print("OR state has",cities['OR'])

#print states
print('-'*10)
print('Michigan`s abbreviation is:',states['Michigan'])

#print all state
print('-'*10)
for state,abbrev in list(states.items()):
    print(f"{state} is abbreviated {abbrev}")
    pass
#print all city
print('-'*10)
for city, abbrev in list(cities.items()):
    print(f"{abbrev} has the city {city}")
    pass

#do both at same time
print("-"*10)
for state,abbrev in list(states.items()):
    print(f"{state} state is abbreviated {abbrev}")
    print(f"and has city {cities[abbrev]}")
    pass

#safety get a abbreviation by state that might not be there
print('-'*10)
state = states.get('Texas')

if not state:
    print("not have texas")
    pass

# get a city with default value
city = cities.get('TX','Do not Exist')
print(f"The city for the state 'TX' is :{city}")
print(cities.items())