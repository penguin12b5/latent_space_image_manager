import imagemanager as im

#setup imagemodifiers
lim = im.LatentImageModifier("images/*.png")
lim.resize_images()
lim.encode_images()

#modify images
final_image = lim.decode_image(lim.modify("none"))


#plot and display images and latent images
ip = im.ImagePlotter(len(lim.images) + 1, 2, 32, 8 * (len(lim.images) + 1))

for i, image in enumerate(lim.images):
    ip.plot_image(image, 2 * i + 1, f'Image {i + 1}')
    ip.plot_image(lim.latent_images[i][0, 0].cpu().numpy(), 2 * i + 2, f'Latent {i + 1}', cmap='viridis')

ip.display_plot()
