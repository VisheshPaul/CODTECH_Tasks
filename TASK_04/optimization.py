import pulp

#problem
model = pulp.LpProblem("Bakery_Profit", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable("Cakes", lowBound=0)
y = pulp.LpVariable("Cookies", lowBound=0)

#Objective function
model += 50 * x + 30 * y

model += 2 * x + y <= 10
model += x + y <= 6

model.solve()

#Result
print("Status:", pulp.LpStatus[model.status])
print("Cakes to produce:", x.varValue)
print("Cookies to produce:", y.varValue)
print("Maximum Profit:", pulp.value(model.objective))
