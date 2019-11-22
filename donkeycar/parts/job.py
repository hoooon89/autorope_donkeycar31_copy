import os
import time
import threading
import math
import shutil


class Job(object):
    def __init__(self, inputs, fn):
        self.inputs = inputs
        self.fn = fn
        self.outputs = None

    def run(self):
        self.outputs = self.fn(*self.inputs)


class JobQueue(object):
    def __init__(self):
        self.jobs = []
        self.lock = threading.Lock()

    def add(self, job):
        with self.lock:
            self.jobs.append(job)

    def get(self):
        with self.lock:
            if len(self.jobs) == 0:
                return None
            job = self.jobs.pop(0)
        return job


class Worker(object):
    def __init__(self, manager):
        self.stop = False
        self.manager = manager        
        self.th = threading.Thread(target=self.update)
        self.th.setDaemon(True)
        self.th.start()

    def update(self):
        while not self.stop:
            job = self.manager.get_queued_job()
            if job:
                job.run()
                self.manager.finished_job(job)
            else:
                time.sleep(0.1)

    def close(self):
        if self.stop is False:
            self.stop = True
            self.th.join()

    def __del__(self):
        self.close()


class JobManager(object):
    '''
    a multi-threaded job manager
    '''
    def __init__(self, num_threads):
        self.workers = []
        self.todo = JobQueue()
        self.finished = JobQueue()
        for _ in range(num_threads):
            worker = Worker(self)
            self.workers.append(worker)

    def queue_job(self, job):
        assert(type(job) is Job)
        return self.todo.add(job)

    def get_queued_job(self):
        return self.todo.get()

    def finished_job(self, job):
        return self.finished.add(job)

    def get_finished_job(self):
        return self.finished.get()

    def close(self):
        for worker in self.workers:
            worker.close()
        self.workers.clear()

    def __del__(self):
        self.close()





def test_job_manager():
   
    tmpDir = "tmp"

    def func_add(a, b):
        return a + b

    def func_calc(a, b):
        res = 0.0
        while a > 0:
            res += math.sin(float(a) / float(b))
            a -= 1
            b -= 1
        return res
    
    def func_calc_and_write_file(a, b):
        fnm = "%s/%d_%d.res" % (tmpDir, a, b)
        res = math.sin(float(a) / float(b))
        with open(fnm, "wt") as outfile:
            outfile.write("%f\n" % res)
        return res

    def func_windows_dir(a, b):
        fnm = "%s\\%d_%d.res" % (tmpDir, a, b)
        return os.system("dir >> %s" % fnm)

    worker_func = func_calc

    inputs = []

    for i in range(10000):
        inputs.append((i, i + 1))
    
    num_jobs = len(inputs)
    num_finished_jobs = 0

    pm = JobManager(num_threads=8)

    os.mkdir(tmpDir)
    s = time.time()
    
    for i in inputs:
        job = Job(inputs=i, fn=worker_func)
        pm.queue_job(job)

    while num_finished_jobs < num_jobs:
        job = pm.get_finished_job()
        if job:
            num_finished_jobs += 1
    
    e = time.time()
    shutil.rmtree(tmpDir)
    print("multi-threaded job finished in", e - s, "seconds.")

    os.mkdir(tmpDir)
    s = time.time()
    for i in inputs:
        _ = worker_func(*i)
    e = time.time()
    print("single threaded job finished in", e - s, "seconds.")
    shutil.rmtree(tmpDir)

    pm.close()

if __name__ == "__main__":
    test_job_manager()
