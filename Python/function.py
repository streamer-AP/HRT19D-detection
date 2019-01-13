def greet_user(username):
    print("hello,"+username)

def greet_user2(username,place="Hit"):
    print("hello,",username,",welcome to",place)

def greet_user3(users):
    for user in users:
        print("hello",user)
    users[0]="HRTer"

users=["Zhao", "Qian", "Sun"]
greet_user3(users)
greet_user3(users)
users[0]="Zhao"
greet_user3(users[:])
greet_user3(users[:])

greet_user("HRTer")
greet_user2(place="Hit",username="HRTer")
greet_user2(username="HRTer")