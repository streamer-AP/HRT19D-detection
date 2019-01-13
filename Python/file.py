with open("testfile.txt","w") as file_object:
    file_object.writelines(["hello\r","HRTER"])

with open("testfile.txt","r") as file_object:
    print(file_object.readlines())
