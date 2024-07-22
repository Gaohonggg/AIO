import numpy as np

def create_train_data():
    data = [["Sunny", "Hot", "High", "Weak", "no"],
            ["Sunny", "Hot", "High", "Strong", "no"],
            ["Overcast", "Hot", "High", "Weak", "yes"],
            ["Rain", "Mild", "High", "Weak", "yes"],
            ["Rain", "Cool", "Normal", "Weak", "yes"],
            ["Rain", "Cool", "Normal", "Strong", "no"],
            ["Overcast", "Cool", "Normal", "Strong", "yes"],
            ["Overcast", "Mild", "High", "Weak", "no"],
            ["Sunny", "Cool", "Normal", "Weak", "yes"],
            ["Rain", "Mild", "Normal", "Weak", "yes"]]
    return np.array(data)

def compute_prior_probability(train_data):
    y_unique = ["no", "yes"]
    prior_probability = np.zeros( len(y_unique) )
    for i in range( len(train_data) ):
        if train_data[i][4] == "no":
            prior_probability[0] = prior_probability[0] + 1
        else:
            prior_probability[1] = prior_probability[1] + 1
    return prior_probability/len( train_data )

def compute_conditional_probability(train_data):
    y_unique = ["no", "yes"]
    conditional_probability = []
    list_x_name = []
    for i in range(0,train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:,i])
        list_x_name.append( x_unique )
        print( x_unique )
        temp = []
        for y in y_unique:
            l = []
            for x in x_unique:
                count = np.sum((train_data[:,i]==x) & (train_data[:,4]==y))
                count_y = np.sum(train_data[:,4]==y)
                l.append(count/count_y)
            temp.extend([l])
        conditional_probability.extend([temp])
    print(conditional_probability)
    return conditional_probability,list_x_name

def get_index_from_value(feature_name, list_features):
    print( np.where(list_features == feature_name) )
    return np.where(list_features == feature_name)[0][0]

train_data = create_train_data()
print( train_data )
prior_probability = compute_prior_probability( train_data )
print("P(play tennis = No) = ",prior_probability[0])
print("P(play tennis = Yes) = ",prior_probability[1])

conditional_probability,list_x_name = compute_conditional_probability(train_data)
outlook = list_x_name[0]

i1 = get_index_from_value("Overcast",outlook)
i2 = get_index_from_value("Rain",outlook)
i3 = get_index_from_value("Sunny",outlook)
print( i1,i2,i3 )
print("P(Outlook = Sunny | Play Tennis = Yes) = ",np.round(conditional_probability[0][1][i3],2))
print("P(Outlook = Sunny | Play Tennis = No) = ",np.round(conditional_probability[0][0][i3],2))

X = ["Sunny","Cool","High","Strong"]
x1 = get_index_from_value(X[0],list_x_name[0])
x2 = get_index_from_value(X[1],list_x_name[1])
x3 = get_index_from_value(X[2],list_x_name[2])
x4 = get_index_from_value(X[3],list_x_name[3])

p0 = (conditional_probability[0][0][x1]
      *conditional_probability[1][0][x2]
      *conditional_probability[2][0][x3]
      *conditional_probability[3][0][x4]
      *prior_probability[0])
p1 = (conditional_probability[0][1][x1]
      *conditional_probability[1][1][x2]
      *conditional_probability[2][1][x3]
      *conditional_probability[3][1][x4]
      *prior_probability[0])
if p0>p1:
    print("Ad should not go!")
else:
    print("Ad should go!")