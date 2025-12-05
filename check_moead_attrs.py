from pymoo.algorithms.moo.moead import MOEAD
import inspect

print("MOEAD init signature:")
print(inspect.signature(MOEAD.__init__))

moead = MOEAD(ref_dirs=[[1.0]], n_neighbors=2, prob_neighbor_mating=0.9)
print("\nMOEAD attributes:")
print(moead.__dict__.keys())

print(f"\nSelection: {moead.selection}")
print(f"Selection attributes: {moead.selection.__dict__}")

