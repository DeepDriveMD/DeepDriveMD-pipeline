from aggregator_reader import *

def test1(ar):
    for j in range(20):
        print('='*30)
        i, steps, md5s, cms, positions, velocities = ar.next_all(n)
        print(f"j={j}, i={i}")
        print(f"md5s = {md5s}")
        print(f"steps = {steps}")

    print(type(positions[0]))
    print(type(cms[0]))
    print(type(md5s[0]))
    print(type(steps[0]))
    print(type(velocities[0]))
    print(positions[0].shape)
    print(cms[0].shape)
    print(velocities[0].shape)

def test2(ar):
    for j in range(7):
        print('*'*30)
        i, cms = ar.next_cm(n)
        print(f"j={j}, i={i}")

    print(type(cms[0]))
    print(cms[0].shape)

def test3():
    s = STREAMS(['aggregator0.bp'])
    z = s.next_cm()
    print(type(z))
    print(z.shape)

def test4():
    s = STREAMS(['aggregator0.bp','aggregator1.bp'])
    z = s.next()
    n = len(z)
    print(n)
    for i in range(n):
        print(i)
        print(type(z[i]))
        print(z[i].shape)

if(__name__ == '__main__'):

    fn = 'aggregator0.bp'
    ADIOS_XML_AGGREGATOR = 'adios.xml'
    stream_name = 'AggregatorOutput'

    ar = ADIOS_READER('aggregator0.bp', 'adios.xml', 'AggregatorOutput')

    test1(ar)
    print("&"*30)
    test2(ar)
    print("&"*30)
    test3()
    print("&"*30)
    test4()
