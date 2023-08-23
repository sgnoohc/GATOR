#ifndef ARGPARSE_H
#define ARGPARSE_H

#include <getopt.h>

class ArgParse
{
private:
    bool checkGraphType(std::string graph)
    {
        if (graph == "T3")
        {
            return true;
        }
        else if (graph == "LS")
        {
            return true;
        }
        else
        {
            return false;
        }
    };

    void help()
    {
        // FIXME
    };

public:
    bool verbose = false;
    bool debug = false;
    std::string graph = "";
    TString tree_name = "tree";
    TString output_file = "output.root";
    std::vector<TString> input_files = {};

    ArgParse() {};

    void parse(int argc, char** argv)
    {
        // Parse CLI input
        while (true) 
        {
            static struct option options[] = {
                {"help", no_argument, 0, 'h'},
                {"verbose", no_argument, 0, 'v'},
                {"debug", no_argument, 0, 'd'},
                {"graph", required_argument, 0, 'g'},
                {"tree", required_argument, 0, 't'},
                {"output_file", required_argument, 0, 'o'},
            };
            int option_index = 0;
            int value = getopt_long(argc, argv, "hvdg:t:o:", options, &option_index);
            if (value == -1) { break; }
            switch(value)
            {
                case 'h':
                    help();
                    exit(EXIT_SUCCESS);
                    break;
                case 'v':
                    verbose = false;
                    break;
                case 'd':
                    debug = false;
                    break;
                case 'g':
                    graph = optarg;
                    break;
                case 't':
                    tree_name = TString(optarg);
                    break;
                case 'o':
                    output_file = TString(optarg);
                    break;
            }
        }

        // Check arguments
        if (graph == "")
        {
            throw std::runtime_error("Error - no graph type specified (use -g or --graph)");
        }
        else if (!checkGraphType(graph))
        {
            throw std::runtime_error("Error - graph type not supported: " + graph);
        }

        // Read all non-optioned arguments as input file paths
        if (optind < argc)
        {
            while (optind < argc)
            {
                input_files.push_back(TString(argv[optind++]));
            }
        }
        if (input_files.size() == 0)
        {
            throw std::runtime_error("Error - no input files specified.");
        }
    };
};

#endif
