# Get input from the user
user_input = input("Enter a string: ")

cleaned_string = ''.join(c for c in user_input.lower() if c.isalnum())

is_palindrome = cleaned_string == cleaned_string[::-1]

if is_palindrome:
    print(f"{user_input} is a palindrome")
else:
    print(f"{user_input} is not a palindrome")