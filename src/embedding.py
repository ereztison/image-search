from PIL import Image
import clip

CLIP_MAX_TEXT_LENGTH = 77


def embed_image(model, preprocess, device, image_path):
    # # random 512 dim vector
    # return np.random.rand(512)

    # preprocess image using CLIP
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)

    # get image embedding
    img_embedding = model.encode_image(img)
    img_embedding /= img_embedding.norm(dim=-1, keepdim=True)

    return img_embedding.cpu().detach().numpy().squeeze()


def embed_text(model, device, text):
    # # random 512 dim vector
    # return np.random.rand(512)

    # tokenize text using CLIP (while ensuring it is not too long)
    text = text[:CLIP_MAX_TEXT_LENGTH]
    text = clip.tokenize(text, context_length=CLIP_MAX_TEXT_LENGTH).to(device)

    # generate text embedding
    text_embedding = model.encode_text(text)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding.cpu().detach().numpy().squeeze()
