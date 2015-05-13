from dataset import dataset

if __name__ == "__main__":
    ds = dataset("data/test.txt", "UTF-8")
    ds.load_triples()
    T = ds.build_csr_matrix()

    print ds

    print "*" * 50
    print "All entities:"
    print ds.get_all_entity() 
    print "-" * 50
    print "All relations:"
    print ds.get_all_relation()
    print "-" * 50
    print "Number of entities"
    print ds.get_num_entity()
    print "-" * 50
    print "Number of relations"
    print ds.get_num_relation()
    print "*" * 50

    print "Tensor matrix:"
    for t in T:
        print t.todense()
    print "-" * 50
    print "*" * 50
