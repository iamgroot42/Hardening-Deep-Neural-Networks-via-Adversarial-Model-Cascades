from keras.models import load_model, Sequential
import sys

x = load_model(sys.argv[1])
y = load_model(sys.argv[2])
z = Sequential()

print x.input, x.output
print y.input, y.output

z.add(x)
z.add(y)

z.save(sys.argv[3])

