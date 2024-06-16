class Student():
    def __init__(self,name,yob,grade):
        self.name = name
        self.yob = yob
        self.grade = grade
        self.job = "Student"
    def describe(self):
        print("Student - Name: {} - Yob: {} - Grade: {}".format(self.name,self.yob,self.grade))

class Teacher():
    def __init__(self,name,yob,subject):
        self.name = name
        self.yob = yob
        self.subject = subject
        self.job = "Teacher"
    def describe(self):
        print("Teacher - Name: {} - Yob: {} - Subject: {}".format(self.name,self.yob,self.subject))

class Doctor():
    def __init__(self,name,yob,specialist):
        self.name = name
        self.yob = yob
        self.specialist = specialist
        self.job = "Doctor"
    def describe(self):
        print("Doctor - Name: {} - Yob: {} - Grade: {}".format(self.name,self.yob,self.specialist))

class Ward():
    def __init__(self,name):
        self.name = name
        self.l = []
    def add_person(self,person):
        self.l.append(person)
    def describe(self):
        for person in self.l:
            person.describe()

    def count_doctor(self):
        count  = 0
        for person in self.l:
            if person.job == "Doctor":
                count += 1
        print("The number of Doctor is: {}".format(count))

    def sort_age(self):
        self.l.sort(key = lambda  x : x.yob,reverse=True)

    def compute_average(self):
        a = []
        for person in self.l:
            if person.job == "Teacher":
                a.append( person.yob )
        print("Average year of birth (teachers): {}".format( sum(a)/len(a)) )
student1 = Student("Nguyen Tat Dat",2004,"12")
student1.describe()

teacher1 = Teacher("Ngoc Diem",1988,"Calculus")
teacher1.describe()

doctor1 = Doctor( "doctorA ",1945 ,"Endocrinologists ")
doctor1.describe()

teacher2 = Teacher( "teacherB ", 1995 , "History ")
doctor2 = Doctor( "doctorB ", 1975 , "Cardiologists ")
ward1 = Ward( " Ward1 ")
ward1.add_person( student1 )
ward1.add_person( teacher1 )
ward1.add_person( teacher2 )
ward1.add_person( doctor1 )
ward1.add_person( doctor2 )
ward1.describe()

ward1.count_doctor()

ward1.sort_age()
ward1.describe()

ward1.compute_average()