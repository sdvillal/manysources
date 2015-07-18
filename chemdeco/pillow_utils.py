# coding=utf-8
from PIL import Image, ImageOps, ImageDraw
from math import ceil
import os.path as op

GRADIENT_FEO = op.join(op.dirname(__file__), 'data', 'gradientfeo.png')


def pil1():
    im = Image.open('/home/santi/example-smartsviewer.ps')
    im.load(scale=2)
    decoborder_size = 50
    im = ImageOps.expand(im, border=decoborder_size, fill='white')
    drawer = ImageDraw.Draw(im)
    drawer.rectangle((525, 13, 600, 48), outline='blue', fill='green')
    drawer.ellipse((525, 13, 600, 48), outline='blue', fill='pink')
    drawer.rectangle((525 + 100, 13, 600 + 100, 48), outline='blue', fill='red')
    drawer.ellipse((525 + 100, 13, 600 + 100, 48), outline='blue', fill='yellow')
    drawer.rectangle((525 - 100, 13, 600 - 100, 48), outline='blue', fill='green')
    drawer.ellipse((525 - 100, 13, 600 - 100, 48), outline='blue', fill='brown')
    im.show()


from rdkit.Chem import AllChem
from rdkit.Chem import Draw


def rdkit2im(mol=None, size=(200, 200)):
    if mol is None:
        nanosmiles = 'CCCC#CC1=CC(=CC(=C1)C#CC2=CC(=C(C=C2C#CC(C)(C)C)C3OCCO3)C#CC(C)(C)C)C#CCCC'
        mol = AllChem.MolFromSmiles(nanosmiles)
    # AllChem.GenerateDepictionMatching2DStructure()
    im = Draw.MolToImage(mol, size=size)
    return im


def artdeco1(im, height=50, decos=(('red', 'blue'),
                                   ('red', 'orange'),
                                   ('green', 'cyan'))):
    # this will add colored markers (circle embeded into a rectangle) in the upper left corner of the picture
    # w, h = im.size
    decos_h = height - 10
    decos_w = decos_h
    im = ImageOps.expand(im, border=height, fill='white')
    drawer = ImageDraw.Draw(im)
    for i, (circc, rectc) in enumerate(decos):
        w_base = 10 + (i * decos_w) + i * 5
        drawer.rectangle((w_base, 5, w_base + decos_w, 5 + decos_h), outline='black', fill=circc)
        drawer.ellipse((w_base, 5, w_base + decos_w, 5 + decos_h), outline='black', fill=rectc)
    return im


def artdeco2(im, chorrada=10, color='red'):
    # this will add a colored border to the image
    return ImageOps.expand(im, border=chorrada, fill=color)


def artdeco3(im, rank=0.5, framesize=30, expand=False, gradient=GRADIENT_FEO):
    # this will add a vertical slider to the right of the picture
    patch = Image.open(gradient) if isinstance(gradient, basestring) else gradient
    patch = patch.resize((int(im.size[0] * 0.03), int(im.size[1] * 0.8)), Image.ANTIALIAS)
    drawer = ImageDraw.Draw(patch)
    drawer.line((0, patch.size[1] * rank, patch.size[0], patch.size[1] * rank), fill='pink', width=5)
    if expand:
        im = artdeco2(im, chorrada=framesize, color='white')
    w, h = im.size
    wp, hp = patch.size
    im.paste(patch, (w - (wp + 10), (h - hp) / 2))
    return im


def artdeco4(im, text):
    # this will add a text at the bottom of the picture
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), text, (0, 0, 0))
    return im


def gridear(ims, num_cols=3):
    # assuming all ims have the same size
    ws = [im.size[0] for im in ims]
    hs = [im.size[1] for im in ims]
    w, h = max(ws), max(hs)
    num_rows = int(ceil(len(ims) / float(num_cols)))
    new_im = Image.new('RGBA', (w * num_cols, h * num_rows), color=(255, 255, 255, 0))
    # even if all ims are not the same size we will not see big black patches
    for i, im in enumerate(ims):
        row = i / num_cols
        col = i % num_cols
        x = col * w
        y = row * h
        new_im.paste(im, (x, y))
    return new_im


if __name__ == '__main__':
    im1 = artdeco2(artdeco1(rdkit2im(), height=30), color='red')
    im2 = artdeco2(artdeco3(artdeco1(rdkit2im(), height=30)), color='green')
    gridear([im1, im2] * 10, num_cols=3).show()