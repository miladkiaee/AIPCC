def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + "/home/milad/PycharmProjects/sync_coin_database/logs/")
    return


import threading

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()