import openai
openai.api_key = '#place your own key here'

response = openai.Image.create(
  prompt="bottle of water",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)