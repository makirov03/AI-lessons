name = "Aysha"
password = "Hajy"

enter_name = input("Enter your name: ")
enter_pass = input("Enter your pass: ")

if enter_name == name and enter_pass == password:
    print("-  -  -\nSuccess!")
else:
    print("-  -  -\nLogin failed!")
