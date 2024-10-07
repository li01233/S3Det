x = torch.randn(4,64,256)
model = SCHEME(embed_dim=256,a=0.7,num_heads=4)
print(model(x).shape)