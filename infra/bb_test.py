from infra.backbone import ResBlock, ResNet



def main():
    rblock = ResBlock(20)
    sdict = rblock.state_dict()
    print(sdict.keys())


    print()

    rnet = ResNet([20, 20, 20])
    sdict_net = rnet.state_dict()
    print(sdict_net.keys())

    print(rnet)

if __name__ == '__main__':
    main()
