import utils.utils as utils
from src.utils import load_data
from models.baseline import bart, pegasus, flan_t5, olmo, DistilBart

if __name__ == "__main__":
    args = utils.get_args()

    data_list = load_data.load_data_from_json(
        args.data_path, args.data_option, args.key_option)
    if args.model == 'bart':
        output, gold_metareview = bart.run_bart_summarization(
            args.sample_size, data_list)
    elif args.model == 'pegasus':
        output, gold_metareview = pegasus.run_pegasus_summarization(
            args.sample_size, data_list)
    # elif args.model == 'pegasus-x':
    #     output, gold_metareview = pegasus_x.run_pegasus_x_summarization(args.sample_size, data_list)
    elif args.model == 'flan-t5':
        output, gold_metareview = flan_t5.run_flan_t5_summarization(
            args.sample_size, data_list)
    elif args.model == 'DistilBart':
        output, gold_metareview = DistilBart.run_distilbart_summarization(
            args.sample_size, data_list)
    elif args.model == 'olmo':
        output, gold_metareview = olmo.run_olmo_summarization(
            args.sample_size, data_list)

    if output is None or gold_metareview is None:
        print("No output or gold meta-review generated.")
        exit(1)

    if not args.output_path:
        if not args.sample_size:
            sample_size_str = "full"
        else:
            sample_size_str = str(args.sample_size)
        # If no output path is provided, save to the default output directory
        args.output_path = f"output/{args.model}_{args.key_option}_{sample_size_str}_out.txt"

    with open(args.output_path, 'w') as f:
        for i in range(len(output)):
            f.write(f"Generated Summary {i+1}:\n")
            f.write(output[i] + "\n")
            # print(output[i])
            f.write(f"Gold Metareview {i+1}:\n")
            f.write(gold_metareview[i] + "\n")
            f.write("\n")
    print(f"Summarization output saved to {args.output_path}")
