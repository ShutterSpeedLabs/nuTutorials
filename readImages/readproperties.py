from nuimages import NuImages

nuim = NuImages(dataroot='/media/parashuram/AutoData/nuImagesMini/', version='v1.0-train', verbose=True, lazy=True)

nuim.category[0]

print(nuim.category[0])

print("table name:", nuim.table_names)
