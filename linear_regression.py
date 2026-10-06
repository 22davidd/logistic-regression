from math import sqrt, exp, log
import time

"""
welcome to my example of logistic regression. today, 6 oct 2026 i am implementing in python what i've learned from the google crash course on machine learning and data science. the model below, trained on 30 entries of data from 2 classes, takes a person's age and their salary and predicts if they would buy an item, lets say a car, why not"""

ages = [18, 22, 30, 47, 38, 25, 34, 41, 29, 52, 19, 27, 36, 45, 31, 23, 39, 48, 26, 33, 21, 44, 37, 55, 28, 42, 35, 50, 24, 40]
salaries = [1000, 1800, 2400, 3200, 2800, 1600, 2600, 3500, 2200, 4100, 1200, 2100, 2900, 3700, 2500, 1500, 3100, 3900, 1900, 2700, 1400, 3600, 3000, 4500, 2300, 3400, 2800, 4200, 1700, 3300]
will_buy = [0,0,1,0,1,0,1,1,0,1,0,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,1,0,1]

# normalization
ages_std, salary_std, ages_mean, salary_mean  = 0, 0, 0, 0
z, w1, b, w2 = 0, 0, 0, 0

# hyperparameters
lr = 0.005
epochs = 100_000


for age in ages:
  ages_mean+=age
ages_mean=ages_mean/len(ages)


for salary in salaries:
  salary_mean+=salary
salary_mean=salary_mean/len(salaries)


for age in ages:
  ages_std+=(ages_mean-age)**2
ages_std=ages_std/len(ages)
ages_std=sqrt(ages_std)

# standard deviations for both categories

for salary in salaries:
  salary_std+=(salary_mean-salary)**2
salary_std=salary_std/len(salaries)
salary_std=sqrt(salary_std)

print("-"*10, "Logistic Regression by 22davidd", "-"*10)
print(f"average age: {ages_mean:.2f}")
print(f"average salary: {salary_mean:.2f}")
print(f"batch size: {len(ages)}")
print(f"epochs: {epochs}")
print(f"learning rate: {lr}")
print("-"*54)
print("\n")
print("the model will now adjust the weights and bias according to the data given at the beginning")
time.sleep(1)
print("\n\n")
print("starting training in ", end="", flush=True)
for i in range(5):
  if i == 4:
    print(f"{5-i} ", end="", flush=True)
  else:
    print(f"{5-i}, ", end="", flush=True)
  time.sleep(1)
time.sleep(0.5)
print("\n")
# sigmoid
def sigmoid(t):
  return 1/(1+exp(-t))

age_nml = []
salary_nml = []

for age in ages:
  age_nml.append((age-ages_mean)/ages_std)  
for salary in salaries:
  salary_nml.append((salary-salary_mean)/salary_std)


# and now the training
for epoch in range(epochs):
  dw1, dw2, db, loss, t_loss= 0, 0, 0, 0, 0

  for t in range(len(ages)):
    x1 = age_nml[t]
    x2 = salary_nml[t]
    y = will_buy[t]
    z = w1*x1 + w2*x2 + b
    p = sigmoid(z)
    dw1 += (p-y)*x1
    dw2 += (p-y)*x2
    db += p-y
    loss = -(y*log(p)+ (1-y)*log(1-p))
    t_loss += loss
    
  dw1, dw2, db = dw1/len(ages), dw2/len(salaries), db/len(ages)
  t_loss = t_loss/len(ages)
  w1 -= lr*dw1
  w2 -= lr*dw2
  b -= lr*db
# printing the loss every 1/10th of the epochs to see the difference
  if(epoch % 10000 == 0):
    print(f"epoch {epoch} - loss: {t_loss:.5f}")

time.sleep(2)
print("\n")
print(f"the weights are now \nw1(age): {w1} \nw2(salary): {w2} \n")
time.sleep(1.2)
print(f"now our test person, Jake")
test_person = [21, 2300]
time.sleep(0.8)
print(f"age: {test_person[0]} \nsalary: {test_person[1]} \n")
normalized_test_person=[]
normalized_test_person.append((test_person[0]-ages_mean)/ages_std)
normalized_test_person.append((test_person[1]-salary_mean)/salary_std)
test_z = w1*normalized_test_person[0] + w2*normalized_test_person[1] + b
time.sleep(1.4)
print(f"the output for Jake: ({w1:.2f})*({normalized_test_person[0]:.2f}) + ({w2:.2f})*({normalized_test_person[1]:.2f}) + {b:.2f}")
time.sleep(1.4)
print(f"which is {test_z:.3f}\n\n")
test_p = sigmoid(test_z)
time.sleep(1.3)
print(f"and now the probability, calculated with the formula 1/(1+exp(-z))")
print(f"1/1+exp({-test_z:.4f}) = {test_p:.4f}\n")
time.sleep(1.2)
print("\n")
# number above which the result matches the positive class
threshold = 0.63
time.sleep(1.3)
print("so, in conclusion")
time.sleep(1.2)
if test_p > threshold:
    print(f"the test person will likely buy the item with a {test_p*100:.2f}% probability")
else:
    print(f"the test person will likely not buy the item, with a {(1-test_p*100):.2f}% probability")

time.sleep(1)
print("nice isn't it? i hope you liked my little code")
print("with love, 22davidd.")










