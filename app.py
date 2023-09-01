from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, redirect
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import DiffusionPipeline
from flask_cors import CORS
from random import shuffle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoTokenizer
from os import urandom
import random
import re
from PIL import Image, ImageDraw, ImageFont
import os
import PIL
import requests
import datetime2
import schedule
import threading
import torch
import io  
# import urllib.parse
# Example using Celery detectors access_token
from celery import Celery


model_nameGPT = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_nameGPT)
model = GPT2LMHeadModel.from_pretrained(model_nameGPT)
#html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_idT2I = "runwayml/stable-diffusion-v1-5"
T2I = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=device, use_safetensors=True, variant="fp16")


model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_device=device, torch_dtype=torch.float32, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

app = Flask(__name__)

# Set the path for profile pictures following.username
profile_pictures_path = os.path.join(app.root_path, 'static', 'profile_pictures')
os.makedirs(profile_pictures_path, exist_ok=True)

# Dictionary to store the user accounts
accounts = {'James': 
                {'password': '1234', 'description': 'Welcome to my profile 🚀', 'email': 'robert@hmail.com', 'profile_picture': None, 'posts': [{'filename': 'IMG_3307.jpeg', 'description': 'A vegetable and delicious land that has been strangely in my dreams for several night 😐😶', 'link': 'https://www.google.com/', 'short_text': 'clik me', 'views': 1200, 'upload_date': '23/08/2023', 'profile_pictures': 'http://localhost:5000/profile_pictures/IMG_3311.jpeg', 'username': 'James', 'ads': False, 'categories': 'Home and decoration', 'verifyed': False, 'urlz': 'eW\x08df]I[dXMVl'}], 'changes': [], 'link': 'www.alapit.com', 'verifyid': False, 'FreeMode': False, 'subscription': False, 'subscription_date': None, 'creation_date': '23/08/2023 20:10:58'},
           }

def encrypt_custom(message, security_key):
    combined_text = message + security_key
    encrypted_text = []
    
    for i, char in enumerate(combined_text):
        shift = ord(security_key[i % len(security_key)])
        encrypted_char = chr((ord(char) + shift))  # Using ASCII values 0-127
        encrypted_text.append(encrypted_char)
        
    return ''.join(encrypted_text)

def decrypt_custom(encrypted_text, security_key):
    decrypted_text = []
    
    for i, char in enumerate(encrypted_text):
        shift = ord(security_key[i % len(security_key)])
        decrypted_char = chr((ord(char) - shift))
        decrypted_text.append(decrypted_char)
        
    return ''.join(decrypted_text)

security_key = "secrtkey"


@app.route('/')
def index():
    return redirect("http://www.example.com", code=302)

@app.route('/login_and_register')
def loginpage():
    return render_template('index.html')


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


@app.route('/create_post', methods=['POST'])
def create_post():
    username = request.form.get('username')
    password = request.form.get('password')
    description = request.form.get('description')
    short_text = request.form.get('short_text')
    link = request.form.get('link')

    if username in accounts and accounts[username]['password'] == password:
        filename = f'{username}_{random.randint(1, 100000)}.png'
        filepath = os.path.join(app.root_path, 'static', 'posts', filename)

        post = {
            'filename': filename,
            'description': description,
            'short_text': short_text,
            'link': link,
            'views': 0,
            'comments': [],
            'profile_pictures': url_for('get_profile_picture', filename=f'{username}.png', _external=True),
            'username': username # Initialize an empty list for comments
        }

        accounts[username]['posts'].append(post)

        # Save the uploaded image to the posts directory
        # (You need to implement the file upload logic)

        return jsonify({'message': 'Post created successfully'})
    else:
        return jsonify({'error': 'Invalid username or password'})



@app.route("/payment-completed", methods=["POST"])
def payment_completed():
    # Trigger your desired event or action here
    # For example: execute_event_on_server()

    return jsonify({"message": "Payment completed and event executed"})




@app.route('/create_account', methods=['POST'])
def create_account():
    username = request.form.get('username')
    password = request.form.get('password')
    description = request.form.get('description')
    email = request.form.get('email')
    link = request.form.get('link')

    if username in accounts:
        return jsonify({'error': 'Username already exists'})

    # Validate email format
    if not validate_email(email):
        return jsonify({'error': 'Invalid email'})

    upload_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")  # Get the current date in the format 'dd/mm/yyyy'
    file = download_image(url_for('get_profile_picture', filename=f'14.jpg', _external=True))
    file.save(os.path.join(profile_pictures_path, f"{username}.png"))

    accounts[username] = {
        'password': password,
        'description': description,
        'email': email,
        'profile_picture': file,
        'posts': [],  # Initialize the list of posts
        'changes': [],  # Initialize the list of changes
        'link': link,
        'verifyid': False,
        'FreeMode': True,
        'subscription': False,
        'subscription_date': None,
        'creation_date': upload_date
    }

    return jsonify({'message': 'Account created successfully'})









@app.route('/operation/<username>', methods=['POST'])
def perform_desired_operation(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]
    # Perform your desired operation using the tx_hash subscription_date
    upload_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")  # Get the current date in the format 'dd/mm/yyyy'

    if user_username in accounts:
        accounts[user_username]['subscription'] = True
        accounts[user_username]['subscription_date'] = upload_date
        print(accounts[user_username])


    response_data = {'message': 'Operation performed successfully'}
    return jsonify(response_data)


















# Function to validate email format
def validate_email(email):
    # Regular expression for email validation
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None


@app.route('/add_comment', methods=['POST'])
def add_comment():
    data = request.json
    username = data.get('username')
    post_filename = data.get('post_filename')
    comment_text = data.get('comment')

    if username and post_filename and comment_text:
        # Check if the username and post exist
        if any(post['filename'] == post_filename for account in accounts.values() for post in account.get('posts', [])):
            # Find the account, post, and comment
            for account in accounts.values():
                for post in account.get('posts', []):
                    if post['filename'] == post_filename:
                        # Add the comment to the post
                        comment = {
                            'text': comment_text,
                            'profile_picture': url_for('get_profile_picture', filename=f'{username}.png', _external=True),  # Use the account's profile picture
                            'username': username  # Use the account's username
                        }
                        post.setdefault('comments', []).append(comment)
                        return jsonify({'success': True, 'message': 'Comment added successfully.', 'comment': comment})

        # Invalid username or post_filename
        return jsonify({'success': False, 'message': 'Invalid username or post_filename.'})

    # Missing required data
    return jsonify({'success': False, 'message': 'Missing required data.'})




@app.route('/search_accounts', methods=['GET'])
def search_accounts():
    keyword = request.args.get('keyword', '').lower()

    matched_accounts = []
    for username, account in accounts.items():
        if keyword in username.lower() or keyword in account['description'].lower():
            matched_posts = []
            for post in account['posts']:
                matched_comments = []
                for comment in post.get('comments', []):
                    comment_with_profile = {
                        'text': comment['text'],
                        'profile_picture': url_for('get_profile_picture', filename=f'{username}.png', _external=True),
                        'username': comment['username']
                    }
                    matched_comments.append(comment_with_profile)

                matched_posts.append({
                    'filename': post['filename'],
                    'description': post['description'],
                    'short_text': post['short_text'],
                    'link': post['link'],
                    'views': post['views'],
                    'profile_picture': url_for('get_profile_picture', filename=f'{username}.png', _external=True),
                    'comments': matched_comments
                })

            matched_accounts.append({
                'username': username,
                'profile_picture': url_for('get_profile_picture', filename=f'{username}.png', _external=True),
                'posts': matched_posts
            })

    return jsonify({'accounts': matched_accounts})



@app.route('/delete_post', methods=['POST'])
def delete_post():
    data = request.get_json()
    filename = data.get('filename')

    # Find the post in the user's account and delete it
    for username, account in accounts.items():
        for post in account['posts']:
            if post['filename'] == filename:
                account['posts'].remove(post)
                # If you want to delete the file from the server as well, use os.remove()
                os.remove(os.path.join(profile_pictures_path, filename))
                return jsonify({'success': True})

    return jsonify({'success': False})




@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.json.get('username')
        password = request.json.get('password')

        if username in accounts and accounts[username]['password'] == password:
            encrypted_message = encrypt_custom(f"{username}&{username}", security_key)
            profile_link = f"/profile/{encrypted_message}"
            return jsonify({'profile_link': profile_link})
        else:
            return jsonify({'error': 'Invalid username or password'})


@app.route('/profile/<username>', methods=['GET', 'POST'])
def profile(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    if user_username in accounts:
        profile = {
            'username': user_username,
            'profile_description': accounts[user_username]['description'],
            'profile_link': accounts[user_username]['link'],
            'profile_picture': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
            'posts': accounts[user_username]['posts'],  # Include the list of posts in the profile data
            'changes': accounts[user_username]['changes'],
            'verifyid': accounts[user_username]['verifyid'],
            'subscription': accounts[user_username]['subscription'],
            'FreeMode': accounts[user_username]['FreeMode'],
            'urlz': username
        }

        user_profile = {
            'username': user_username,
            'profile_picture': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
        }

        if request.method == 'POST' and user_username == request.headers.get('Username'):
            if 'file' in request.files:
                # Handle profile picture upload
                file = request.files['file']
                if file:
                    # Save the file to the server
                    update_profile_picture(user_username, file)

            description = request.form.get('description')
            if description:
                # Update the description and add it to the list of changes
                accounts[user_username]['description'] = description
                accounts[user_username]['changes'].append(f"Description changed to '{description}'")

        # Shuffle the posts for the current account
        random.shuffle(profile['posts'])
        

        # Search functionality
        keyword = request.args.get('keyword', '')
        if keyword:
            matched_accounts = []
            for acc_username, account in accounts.items():
                if keyword.lower() in acc_username.lower() or keyword.lower() in account['description'].lower():
                    matched_accounts.append({
                        'username': acc_username,
                        'profile_picture': url_for('get_profile_picture', filename=f'{acc_username}.png', _external=True)
                    })

            return render_template('profile.html', profile=profile, keyword=keyword, matched_accounts=matched_accounts)
        

        if 'following' in accounts[user_username]:
            following_accounts = accounts[user_username]['following']
            for following in following_accounts:
                if following['username'] == profile_username:
                    profile['is_following'] = True
                    break

        if 'followers' in accounts[profile_username]:
            profile['followers'] = accounts[profile_username]['followers']

        if 'following' in accounts[user_username]:
            following_accounts = accounts[user_username]['following']
            following_profiles = []
            for following in following_accounts:
                following_username = following['username']
                if following_username in accounts:
                    following_profile = {
                        'username': following_username,
                        'profile_picture': url_for('get_profile_picture', filename=f'{following_username}.png', _external=True),
                        'posts': accounts[following_username]['posts'],
                        'urlz': username
                    }
                    following_profiles.append(following_profile)
            user_profile['following'] = following_profiles


        return render_template('profile.html', profile=profile, accounts=accounts, profiles2=user_profile)  # Pass the accounts dictionary to the template filename
 
    return jsonify({'error': 'Profile not found'}), 404




@app.route('/profile_view/<username>', methods=['GET'])
def profile_view(username):

    if username in accounts:
        profile = {
            'username': username,
            'description': accounts[username]['description'],
            'profile_picture': url_for('get_profile_picture', filename=f'{username}.png', _external=True),
            'posts': accounts[username]['posts'],
            'changes': accounts[username]['changes'],  # Include the list of changes in the profile data
            'link': accounts[username]['link']
        }

        return render_template('profile_view.html', profile=profile)

    return jsonify({'error': 'Profile not found'}), 404



@app.route('/profiles/<username>', methods=['GET'])
def profiles(username):
    splitUsernames = username.split("&")
    profile_usernames = splitUsernames[1]
    user_usernames = splitUsernames[0]
    decrypted_message = decrypt_custom(user_usernames, security_key).replace(security_key, "")
    print(decrypted_message)

    finall_text = f"{decrypted_message}&{profile_usernames}"

    splitUsername = finall_text.split("&")
    profile_username = splitUsername[2]
    user_username = splitUsername[0]

    if profile_username in accounts:
        profile = {
            'username': profile_username,
            'description': accounts[profile_username]['description'],
            'profile_picture': url_for('get_profile_picture', filename=f'{profile_username}.png', _external=True),
            'posts': accounts[profile_username]['posts'],
            'changes': accounts[profile_username]['changes'],  # Include the list of changes in the profile data
            'is_following': False,  # Add a flag to indicate if the user is being followed
            'profile_link': accounts[profile_username]['link'],
            'verifyid': accounts[profile_username]['verifyid'],
            'followers': [] # Initialize the followers list
        }

        if user_username in accounts:
            user_profile = {
                'username': user_username,
                'profile_picture': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
            }

            if 'following' in accounts[user_username]:
                following_accounts = accounts[user_username]['following']
                for following in following_accounts:
                    if following['username'] == profile_username:
                        profile['is_following'] = True
                        break

            if 'followers' in accounts[profile_username]:
                profile['followers'] = accounts[profile_username]['followers']

            if 'following' in accounts[user_username]:
                following_accounts = accounts[user_username]['following']
                following_profiles = []
                for following in following_accounts:
                    following_username = following['username']
                    if following_username in accounts:
                        following_profile = {
                            'username': following_username,
                            'profile_picture': url_for('get_profile_picture', filename=f'{following_username}.png', _external=True)
                        }
                        following_profiles.append(following_profile)
                user_profile['following'] = following_profiles

        return render_template('profiles.html', profile=profile, profiles2=user_profile)

    return jsonify({'error': 'Profile not found'}), 404





@app.route('/follow/<username>', methods=['POST'])
def follow(username):

    splitUsernames = username.split("&")
    profile_usernames = splitUsernames[1]
    user_usernames = splitUsernames[0]
    decrypted_message = decrypt_custom(user_usernames, security_key).replace(security_key, "")
    print(decrypted_message)

    finall_text = f"{decrypted_message}&{profile_usernames}"

    splitUsername = finall_text.split("&")
    follower_username = splitUsername[0]
    following_username = splitUsername[2]

    if follower_username in accounts and following_username in accounts:
        if 'followers' not in accounts[following_username]:
            accounts[following_username]['followers'] = []
        followers = accounts[following_username]['followers']
        for follower in followers:
            if follower['username'] == follower_username:
                return jsonify({'error': 'Already following'}), 400

        followers.append({'username': follower_username, 'profile_picture': url_for('get_profile_picture', filename=f'{follower_username}.png', _external=True)})

        if 'following' not in accounts[follower_username]:
            accounts[follower_username]['following'] = []
        following = accounts[follower_username]['following']
        for followed in following:
            if followed['username'] == following_username:
                return jsonify({'error': 'Already following'}), 400

        following.append({'username': following_username, 'profile_picture': url_for('get_profile_picture', filename=f'{following_username}.png', _external=True)})

        return jsonify({'success': True})

    return jsonify({'error': 'Unable to follow'}), 400







@app.route('/unfollow', methods=['POST'])
def unfollow():
    data = request.json
    follower_username = data.get('follower_username')
    following_username = data.get('following_username')

    if follower_username in accounts and following_username in accounts:
        followers = accounts[following_username]['followers']
        for follower in followers:
            if follower['username'] == follower_username:
                followers.remove(follower)
                break

        following = accounts[follower_username]['following']
        for followed in following:
            if followed['username'] == following_username:
                following.remove(followed)
                break

        return jsonify({'success': True})

    return jsonify({'error': 'Unable to unfollow'}), 400



@app.route('/update_profile_picture/<username>', methods=['POST'])
def update_profile_picture(username):

    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    if user_username in accounts:
        file = request.files['file']
        if file:
            # Save the file to the server
            file.save(os.path.join(profile_pictures_path, f"{user_username}.png"))
            return jsonify({'message': 'Profile picture updated successfully'})
        else:
            return jsonify({'error': 'No file uploaded'})
    else:
        return f"No profile found for {user_username}"




@app.route('/update_profile/<username>', methods=['POST'])
def update_profile(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    if user_username in accounts:
        description = request.form.get('description')
        if description:
            # Update the description and add it to profiles2 list of changes
            accounts[user_username]['description'] = description
            accounts[user_username]['changes'].append(description)
            return jsonify({'message': 'Profile updated successfully'})
        else:
            return jsonify({'error': 'Invalid description'})
    else:
        return jsonify({'error': f"No profile found for {user_username}"})



@app.route('/update_username/<username>', methods=['POST'])
def update_username(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    new_username = request.form.get('new_username')
    if new_username:
        if new_username in accounts:
            return jsonify({'error': 'Username already exists'})
        else:
            accounts[new_username] = accounts.pop(user_username)
            return jsonify({'message': 'Username updated successfully'})
    else:
        return jsonify({'error': 'Invalid new username'})



@app.route('/update_link/<username>', methods=['POST'])
def update_link(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    new_link = request.form.get('new_link')
    if new_link:
        accounts[user_username]['link'] = new_link
        return jsonify({'message': 'Link updated successfully'})
    else:
        return jsonify({'error': 'Invalid new Link'})


@app.route('/increment_views/<username>/<post_index>', methods=['GET'])
def increment_views(username, post_index):

    if username in accounts and int(post_index) < len(accounts[username]['posts']):
        accounts[username]['posts'][int(post_index)]['views'] += 1
        return jsonify({'message': 'Views incremented successfully'})
    else:
        return jsonify({'error': 'Invalid username or post index'}), 404







@app.route('/repiting/<username>', methods=['POST'])
def upload_photos(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    if user_username in accounts:
        files = request.form.get('file')
        file = download_image(files)
        description = request.form.get('description')
        link = request.form.get('link')  # Retrieve the link from the request form
        short_text = request.form.get('short_text')  # Retrieve the short text from the request form

        if file:
            file.save(os.path.join(profile_pictures_path, f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg"))
            accounts[user_username]['posts'].append({
                'filename': f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg",
                'description': description,
                'link': link,  # Include the link in the post data
                'short_text': short_text,
                'views': 0,
                'profile_pictures': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
                'username': user_username,  # Include the short text in the post data
                'urlz': username
            })
            # Find the post by its filename
            for post in accounts[user_username]['posts']:
                if post['filename'] == f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg":
                    post['views'] += 1  # Increment the number of views

            # Return the response indicating successful upload
            return jsonify({'message': 'Post uploaded successfully'})
        else:
            return jsonify({'error': 'No file uploaded'})
    else:
        return f"No profile found for {user_username}"


@app.route('/generate_photos/<username>', methods=['POST'])
def generate_photos(username):
    prompt = request.form.get('prompt')
    files = request.form.get('file')
    
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    image = download_image(files)
    image = image.resize((450, 450), Image.ANTIALIAS)  # Resize the image to 450 x 450

    file = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]

    if file:
        filename = f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg"
        file_path = os.path.join(profile_pictures_path, filename)
        file.save(os.path.join(profile_pictures_path, filename))

        # Open the image using PIL
        image = Image.open(file_path)
            
        # Get the default font size of the image
        font_size = 55  # Increase font size by 25 pixels
            
        # Load the default font
        font = ImageFont.load_default()
            
        # Define the watermark text
        watermark_text = f"@{user_username} of alapit.com\nstable-diffusion-v1-5\nand instruct-pix2pix"
            
        # Create a drawing object
        draw = ImageDraw.Draw(image)
            
        # Calculate watermark position
        watermark_position = (10, image.height - font_size - 10)  # Adjust the values as needed
            
        # Calculate background rectangle dimensions
        text_width, text_height = draw.textsize(watermark_text, font=font)
            
        # Draw the watermark text on top of the background
        draw.text(watermark_position, watermark_text, fill=(255, 255, 255), font=font)
            
        # Save the image with the watermark
        image.save(file_path)

        return jsonify({'filename': url_for('get_profile_picture', filename=filename, _external=True)})
    else:
        return jsonify({'error': 'Failed to generate photo'})






@app.route('/upload_photo/<username>', methods=['POST'])
def upload_photo(username):

    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    if user_username in accounts:
        files = request.form.get('file')
        file = download_image(url_for('get_profile_picture', filename=f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg", _external=True))
        description = request.form.get('description')
        link = request.form.get('link')  # Retrieve the link from the request form
        short_text = request.form.get('short_text')  # Retrieve the short text from the request form
        categories = request.form.get('categories')

        if file:
            file.save(os.path.join(profile_pictures_path, f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg"))
            accounts[user_username]['posts'].append({
                'filename': f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg",
                'description': description,
                'link': link,  # Include the link in the post data
                'short_text': short_text,
                'views': 0,
                'profile_pictures': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
                'username': user_username,  # Include the short text in the post data
                'verifyed': accounts[user_username]['verifyid'],
                'categories': categories
            })
            # Find the post by its filename
            for post in accounts[user_username]['posts']:
                if post['filename'] == f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg":
                    post['views'] += 1  # Increment the number of views

            # Return the response indicating successful upload
            return jsonify({'message': 'Post uploaded successfully'})
        else:
            return jsonify({'error': 'No file uploaded'})
    else:
        return f"No profile found for {user_username}"






@app.route('/generate_photo/<username>', methods=['POST'])
def generate_photo(username):
    prompt = request.form.get('prompt')
    files = request.files['file']
    
    if files:
        files.save(os.path.join(profile_pictures_path, files.filename))
    
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    profile_picture_filename = files.filename
    profile_picture_path = os.path.join(profile_pictures_path, profile_picture_filename)
    if os.path.exists(profile_picture_path):
        image = download_image(url_for('get_profile_picture', filename=profile_picture_filename, _external=True))
        image = image.resize((150, 150), Image.ANTIALIAS)  # Resize the image to 450 x 450

        file = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]

        if file:
            filename = f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg"
            file.save(os.path.join(profile_pictures_path, filename))
            return jsonify({'filename': url_for('get_profile_picture', filename=filename, _external=True)})
        else:
            return jsonify({'error': 'Failed to generate photo'})

    return jsonify({'error': 'No profile picture found'})











@app.route('/upload_ads/<username>', methods=['POST'])
def upload_ads(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]
    if user_username in accounts:
        file = download_image(url_for('get_profile_picture', filename=f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg", _external=True))
        description = request.form.get('description')
        link = request.form.get('link')  # Retrieve the link from the request form
        short_text = request.form.get('short_text')  # Retrieve the short text from the request form

        if file:
            file.save(os.path.join(profile_pictures_path, f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg"))
            upload_date = datetime.datetime.now().strftime("%d/%m/%Y")  # Get the current date in the format 'dd/mm/yyyy'
            accounts[user_username]['posts'].append({
                'filename': f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg",
                'description': description,
                'link': link,  # Include the link in the post data
                'short_text': short_text,
                'views': 0,
                'upload_date': upload_date,  # Include the upload date in the post data
                'profile_pictures': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
                'username': user_username,  # Include the short text in the post data
                'ads': True,
                'verifyed': accounts[user_username]['verifyid'],
            })
            # Find the post by its filename
            for post in accounts[user_username]['posts']:
                if post['filename'] == f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg":
                    post['views'] += 1  # Increment the number of views

            

            # Return the response indicating successful upload
            return jsonify({'message': 'Post uploaded successfully'})
        else:
            return jsonify({'error': 'No file uploaded'})
    else:
        return f"No profile found for {user_username}"

#profiles2
@app.route('/generate_ads/<username>', methods=['POST'])
def generate_ads(username):

    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    file = request.files['file']

    if file:
        filename = f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg"
        file.save(os.path.join(profile_pictures_path, filename))
        return jsonify({'filename': url_for('get_profile_picture', filename=filename, _external=True)})
    else:
        return jsonify({'error': 'Failed to generate photo'})























@app.route('/upload_photo2/<username>', methods=['POST'])
def upload_photo2(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]
    if user_username in accounts:
        file = download_image(url_for('get_profile_picture', filename=f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg", _external=True))
        description = request.form.get('description')
        link = request.form.get('link')  # Retrieve the link from the request form
        short_text = request.form.get('short_text')  # Retrieve the short text from the request form
        categories = request.form.get('categories')
        print(categories)

        if file:
            file.save(os.path.join(profile_pictures_path, f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg"))
            upload_date = datetime.datetime.now().strftime("%d/%m/%Y")  # Get the current date in the format 'dd/mm/yyyy'
            accounts[user_username]['posts'].append({
                'filename': f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg",
                'description': description,
                'link': link,  # Include the link in the post data
                'short_text': short_text,
                'views': 0,
                'upload_date': upload_date,  # Include the upload date in the post data
                'profile_pictures': url_for('get_profile_picture', filename=f'{user_username}.png', _external=True),
                'username': user_username,  # Include the short text in the post data
                'ads': False,
                'categories': categories,
                'verifyed': accounts[user_username]['verifyid'],
                'urlz': username
            })
            # Find the post by its filename
            for post in accounts[user_username]['posts']:
                if post['filename'] == f"{user_username}_post_{len(accounts[user_username]['posts']) + 1}.jpg":
                    post['views'] += 1  # Increment the number of views

            
            
            # Return the response indicating successful upload
            return jsonify({'message': 'Post uploaded successfully'})
        else:
            return jsonify({'error': 'No file uploaded'})
    else:
        return f"No profile found for {user_username}"



# @app.route('/generate_photo2/<username>', methods=['POST'])
# def generate_photo2(username):
#     decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
#     splitUsername = decrypted_message.split("&")
#     profile_username = splitUsername[1]
#     user_username = splitUsername[0]

#     file = request.files['file']

#     if file:
#         filename = f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg"
#         file_path = os.path.join(profile_pictures_path, filename)
        
#         # Save the uploaded file
#         file.save(file_path)
        
#         # Open the image using PIL
#         image = Image.open(file_path)
        
#         # Get the default font size of the image
#         font_size = 55  # Increase font size by 25 pixels
        
#         # Load the default font
#         font = ImageFont.load_default()
        
#         # Define the watermark text
#         watermark_text = f"@{user_username}\nGenerated by\nstable-diffusion-v1-5\nand instruct-pix2pix"
        
#         # Create a drawing object
#         draw = ImageDraw.Draw(image)
        
#         # Calculate watermark position
#         watermark_position = (10, image.height - font_size - 10)  # Adjust the values as needed
        
#         # Calculate background rectangle dimensions
#         text_width, text_height = draw.textsize(watermark_text, font=font)
#         background_rectangle = [(watermark_position[0], watermark_position[1]),
#                                 (watermark_position[0] + text_width + 10, watermark_position[1] + text_height)]
        
#         # Draw slightly transparent black rectangle as background
#         draw.rectangle(background_rectangle, fill=(0, 0, 0, 50))  # Adjust alpha value for transparency
        
#         # Draw the watermark text on top of the background
#         draw.text(watermark_position, watermark_text, fill=(255, 255, 255), font=font)
        
#         # Save the image with the watermark
#         image.save(file_path)
        
#         return jsonify({'filename': url_for('get_profile_picture', filename=filename, _external=True)})
#     else:
#         return jsonify({'error': 'Failed to generate photo'})


    

@app.route('/generate_photo2/<username>', methods=['POST'])
def generate_photo2(username):
    decrypted_message = decrypt_custom(username, security_key).replace(security_key, "")
    splitUsername = decrypted_message.split("&")
    profile_username = splitUsername[1]
    user_username = splitUsername[0]

    prompt = request.form.get('prompt')

    profile_picture_filename = f"{user_username}.png"
    profile_picture_path = os.path.join(profile_pictures_path, profile_picture_filename)
    
    if os.path.exists(profile_picture_path):

        file = T2I(prompt).images[0]

        if file:
            filename = f"{user_username}_generated_{len(accounts[user_username]['posts']) + 1}.jpg"
            file_path = os.path.join(profile_pictures_path, filename)
            file.save(os.path.join(profile_pictures_path, filename))
            
            # Open the image using PIL
            image = Image.open(file_path)
            
            # Get the default font size of the image
            font_size = 55  # Increase font size by 25 pixels
            
            # Load the default font
            font = ImageFont.load_default()
            
            # Define the watermark text
            watermark_text = f"@{user_username} of alapit.com\nstable-diffusion-v1-5\nand instruct-pix2pix"
            
            # Create a drawing object
            draw = ImageDraw.Draw(image)
            
            # Calculate watermark position
            watermark_position = (10, image.height - font_size - 10)  # Adjust the values as needed
            
            # Calculate background rectangle dimensions
            text_width, text_height = draw.textsize(watermark_text, font=font)
            
            # Draw the watermark text on top of the background
            draw.text(watermark_position, watermark_text, fill=(255, 255, 255), font=font)
            
            # Save the image with the watermark
            image.save(file_path)

            return jsonify({'filename': url_for('get_profile_picture', filename=filename, _external=True)})
        else:
            return jsonify({'error': 'Failed to generate photo'})

    return jsonify({'error': 'No profile picture found'})


posts_path = os.path.join(app.root_path, 'static', 'posts')


@app.route('/get_preview/<filename>')
def get_preview(filename):
    return send_from_directory(posts_path, filename)

    

@app.route('/profile_pictures/<filename>')
def get_profile_picture(filename):
    return send_from_directory(profile_pictures_path, filename)






@app.route('/admin/accounts', methods=['GET'])
def admin_accounts():
    return render_template('admin_accounts.html', accounts=accounts)


@app.route('/admin/delete_account/<username>', methods=['POST'])
def admin_delete_account(username):
    if username in accounts:
        # Delete the user's posts
        for post in accounts[username]['posts']:
            post_filename = post['filename']
            if os.path.exists(os.path.join(profile_pictures_path, post_filename)):
                os.remove(os.path.join(profile_pictures_path, post_filename))
        # Delete the account
        del accounts[username]
        return jsonify({'message': f'Account {username} and associated posts deleted successfully'})
    else:
        return jsonify({'error': f'Account {username} not found'})




@app.route('/admin/edit_account/<username>', methods=['POST'])
def admin_edit_account(username):
    if username in accounts:
        data = request.json
        if 'description' in data:
            accounts[username]['description'] = data['description']
        if 'email' in data:
            accounts[username]['email'] = data['email']
        if 'link' in data:
            accounts[username]['link'] = data['link']
        # Update other fields as needed
        return jsonify({'message': f'Account {username} updated successfully'})
    else:
        return jsonify({'error': f'Account {username} not found'})





@app.route('/verify_account/<username>', methods=['GET', 'POST'])
def verify_account(username):
    if username in accounts:
        if request.method == 'POST':
            accounts[username]['verifyid'] = True
            return jsonify({'message': f'Account {username} verified successfully'})
        else:
            return render_template('verify_account.html', username=username)  # Create a verification template or redirect to a page
    else:
        return jsonify({'error': 'Account not found'})


@app.route('/admin/toggle_subscription/<username>', methods=['POST'])
def toggle_subscription(username):
    if username in accounts:
        accounts[username]['subscription'] = True
        accounts[username]['subscription_date'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        return jsonify({'message': f'Subscription status for account {username} toggled'})
    else:
        return jsonify({'error': f'Account {username} not found'})

@app.route('/admin/toggle_Unsubcription/<username>', methods=['POST'])
def toggle_Unsubcription(username):
    if username in accounts:
        accounts[username]['subscription'] = False
        accounts[username]['subscription_date'] = None
        
        return jsonify({'message': f'Unsubcription status for account {username} toggled'})
    else:
        return jsonify({'error': f'Account {username} not found'})



# # Define a function to delete old ads
# def delete_old_ads():
#     for username, account in accounts.items():
#         for post in account['posts']:
#             if post['ads']:
#                 upload_date = datetime.datetime.strptime(post['upload_date'], "%d/%m/%Y")
#                 current_date = datetime.datetime.now()
#                 delta = current_date - upload_date
#                 if delta.seconds >= 60:  # Check if the post is older than 1 minute  "if delta.days >= 30:"
#                     post_filename = post['filename']
#                     file_path = os.path.join(profile_pictures_path, post_filename)
#                     if os.path.exists(file_path):
#                         os.remove(file_path)
#                     account['posts'].remove(post)  # Remove the post from the account's posts list

# # Schedule the task to run every minute for testing purposes
# schedule.every(1).minutes.do(delete_old_ads)

# # Define a function to run the scheduled tasks
# def run_schedule():
#     while True:
#         schedule.run_pending()
#         

# # Start the scheduling thread
# schedule_thread = threading.Thread(target=run_schedule)
# schedule_thread.start()

def encrypt(word, shift):
    encrypted_word = ""
    for char in word:
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            encrypted_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            encrypted_word += encrypted_char
        else:
            encrypted_word += char
    return encrypted_word


def decrypt(encrypted_word, shift):
    decrypted_word = ""
    for char in encrypted_word:
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            decrypted_char = chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
            decrypted_word += decrypted_char
        else:
            decrypted_word += char
    return decrypted_word


def generate_text(subject, length=100, temperature=1.0):
    # Prompt the user for the subject of the text

    # Encode the subject text using GPT-2 tokenizer
    input_ids = tokenizer.encode(subject, return_tensors='pt')

    # Generate text using GPT-2 model
    output = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids[0]) + length,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=10
    )

    # Decode generated text using GPT-2 tokenizer and return as string
    generated_texts = []
    for i in range(len(output)):
        generated_texts.append(tokenizer.decode(output[i], skip_special_tokens=True))

    # Select the best generated text
    best_generated_text = max(generated_texts, key=lambda text: len(text.split()))

    # Check if the generated text is about the correct subject
    if subject in best_generated_text:
        return best_generated_text
    else:
        # If the generated text is not about the correct subject, generate a new text
        return generate_text(subject, length, temperature)


def imageTotext(imageLink, length):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    wordnum = int(length)

    # decoded_url = urllib.parse.unquote(imageLink)
    decoded_url = download_image(imageLink)
    t = predict_step([decoded_url], feature_extractor, tokenizer, model, device, gen_kwargs)

    subject = f"This image is {t[0]}"
    IMGT = generated_text = generate_text(subject, length=wordnum, temperature=0.7)
    return f"{IMGT}"

def predict_step(image_paths, feature_extractor, tokenizer, model, device, gen_kwargs):
    images = []
    for image_path in image_paths:
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images.append(image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# <img src="http://localhost:5000/profile_pictures/James_generated_2.jpg" id="generatedImage">

@app.route('/alapit_continuator', methods=['POST'])
def alapit_continuator():
    data = request.get_json()
    inputM = str(data['input'])

    result = imageTotext(inputM, 12)
    return jsonify({'result': result})


# Define a function to delete old ads
def delete_FreeMode():
    for username, account in accounts.items():
        if account['FreeMode']:
            upload_date = datetime.datetime.strptime(account['creation_date'], "%d/%m/%Y %H:%M:%S")
            current_date = datetime.datetime.now()
            delta = current_date - upload_date
            if delta.seconds >= 60:  # Check if the post is older than 1 minute  "if delta.days >= 30:"
                account['FreeMode'] = False
                print(f"{account} has false")


# Schedule the task to run every minute for testing purposes

# Define a function to run the scheduled tasks
def run_schedule():
    while True:
        schedule.run_pending()
        


# Define a function to delete old ads
def delete_operation():
    for username, account in accounts.items():
        if account['subscription']:
            upload_date = datetime.datetime.strptime(account['subscription_date'], "%d/%m/%Y %H:%M:%S")
            current_date = datetime.datetime.now()
            delta = current_date - upload_date
            if delta.seconds >= 120:  # Check if the post is older than 1 minute  "if delta.days >= 30:"
                account['subscription'] = False
                account['subscription_date'] = None
                print(f"{account} has false")


# Schedule the task to run every minute for testing purposes


# Define a function to run the scheduled tasks
def run_schedule2():
    while True:
        schedule.run_pending()
        
# Start the scheduling thread


schedule.every(1).minutes.do(delete_FreeMode)
schedule.every(1).minutes.do(delete_operation)

# Start the scheduling thread
schedule_thread = threading.Thread(target=run_schedule)
schedule_thread.start()
schedule_thread2 = threading.Thread(target=run_schedule2)
schedule_thread2.start()
# host="localhost", port=5000

if __name__ == '__main__':
    app.run()
