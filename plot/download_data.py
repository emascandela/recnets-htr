import wandb
key = "e3dd99069f49577edaf1c80982c52f12fcdebb36"
api = wandb.Api(api_key=key)

# run is specified by <entity>/<project>/<run_id>
# run = api.run("<entity>/<project>/<run_id>")

for exp_name in ['washington', 'saint_gall', 'parzival']:
    out_name = f"{exp_name}2.dat"
    runs = api.runs(f"emascandela/CRNN - {exp_name.upper()}", order="+created_at")
    print(len(runs))

    # run = runs[10]

    runs_dict = {}

    for run in runs:
        # print(run.summary["params"], run.summary["cer"])

        runs_dict[run.name] = run.summary

    with open(out_name, "w") as of:
        outputs = []
        names = []
        q8_cer = {}
        q1_cer = {}

        for name, d in runs_dict.items():
            print(name)
            if name.startswith("*"):
                print("Base")
                cl = 2
            elif name.startswith("#"):
                print("Rec")
                cl = 1
            else:
                print("Cluster")
                cl = 0

            cer = d["cer"] if "cer" in d else -1
            if name.endswith("Q8"):
                baseline_name = name.replace("_Q8", "")
                params = runs_dict[baseline_name]["params"]
                if baseline_name in q8_cer:
                    raise Exception()
                q8_cer[baseline_name] = cer
                continue
            elif name.endswith("Q1.58"):
                baseline_name = name.replace("_Q1.58", "")
                params = runs_dict[baseline_name]["params"]
                if baseline_name in q1_cer:
                    raise Exception()
                q1_cer[baseline_name] = cer
                continue
            
            params = d["params"]
            
            # of.write(f"{params} {cer*100} {cl} {name.split(' ')[-1]}\n")
            outputs.append([params, cer*100, cl, name.split(' ')[-1]])
            names.append(name)
        
        for output, name in zip(outputs, names):
            q8 = (q8_cer[name] * 100) if name in q8_cer else -1
            q1 = (q1_cer[name] * 100) if name in q1_cer else -1
            of.write(" ".join(map(str, output+[q8, q1])) + "\n")
        

            
            
