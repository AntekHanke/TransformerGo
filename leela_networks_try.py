from lczero.backends import Weights, Backend, GameState

w = Weights()
print(w.filters())

print(Backend.available_backends())

b = Backend(weights=w)
g = GameState(moves=['e2e4', 'e7e5'])
print(g.as_string())
i1 = g.as_input(b)
i2 = GameState(fen='2R5/5kpp/4p3/p4p2/3B4/1K5N/4rNPP/8 b - - 0 29').as_input(b)
o1, o2 = b.evaluate(i1, i2)

print(list(zip(g.moves(), o1.p_softmax(*g.policy_indices()))))

print(f"Value: {o2.q()}")

