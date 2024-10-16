// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2024 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <vector>

#include <surface_mesh/Surface_mesh.h>
#include "otsolver_2dgrid.h"
#include "common/otsolver_options.h"
#include "utils/rasterizer.h"
#include "common/image_utils.h"
#include "common/generic_tasks.h"

using namespace Eigen;
using namespace surface_mesh;
using namespace otmap;

void output_usage()
{
    std::cout << "usage : transportfield <option> <value>" << std::endl;

    std::cout << "input options:" << std::endl;
    std::cout << " * -in i0 i1 where i* is either a <filename> or a procedural func \":id:res:\"" << std::endl;
    std::cout << " *           see analytical_functions.h for a list of possible function ids." << std::endl;

    std::cout << "options :" << std::endl;
    std::cout << " * -img_eps <value>    (default: 1e-5)" << std::endl;
    std::cout << " * -inv                use 1-img" << std::endl;
    std::cout << " * -export_maps        write maps as .off files" << std::endl;

    CLI_OTSolverOptions::print_help();

    std::cout << "output options :" << std::endl;
    std::cout << " * -out <prefix>" << std::endl;
}

struct CLIopts : CLI_OTSolverOptions
{
    std::vector<std::string> inputs;
    std::string out_prefix;

    double img_eps;
    bool use_inv;
    bool export_maps;

    void set_default()
    {
        inputs.clear();

        out_prefix = "";
        img_eps = 1e-5;
        use_inv = false;
        export_maps = false;

        CLI_OTSolverOptions::set_default();
    }

    bool load(const InputParser &args)
    {
        set_default();

        CLI_OTSolverOptions::load(args);

        std::vector<std::string> value;

        if (args.getCmdOption("-in", value))
        {
            inputs = value;
        }
        else
        {
            std::cerr << "missing -in filename or -in :id:res: to specify input" << std::endl;
            return false;
        }

        if (args.getCmdOption("-img_eps", value))
            img_eps = std::stof(value[0]);

        if (args.getCmdOption("-out", value))
            out_prefix = value[0];

        if (args.cmdOptionExists("-inv"))
            use_inv = true;

        if (args.cmdOptionExists("-export_maps"))
            export_maps = true;

        return true;
    }
};

void compute_transport_field(const Surface_mesh &start, const Surface_mesh &end, std::vector<Eigen::Vector2d> &vfield);

std::string
make_padded_string(int n, int nzero = 3)
{
    std::stringstream ss;
    ss << "_";
    ss << std::setfill('0') << std::setw(nzero) << n;
    return ss.str();
}

int main(int argc, char **argv)
{
    InputParser input(argc, argv);

    if (input.cmdOptionExists("-help") || input.cmdOptionExists("-h"))
    {
        output_usage();
        return 0;
    }

    CLIopts opts;
    if (!opts.load(input))
    {
        std::cerr << "invalid input" << std::endl;
        output_usage();
        return 1;
    }

    if (opts.inputs.size() != 2)
    {
        std::cerr << "Got " << opts.inputs.size() << " input densities, but you must provide 2.\n";
        output_usage();
        return 1;
    }

    std::vector<TransportMap> tmaps;
    std::vector<MatrixXd> input_densities;
    generate_transport_maps(opts.inputs, tmaps, opts,
                            [&opts, &input_densities](MatrixXd &img)
                            {
                                if (opts.use_inv)
                                    img = 1. - img.array();

                                img.array() += opts.img_eps;

                                input_densities.push_back(img);
                            });

    std::vector<Surface_mesh> inv_maps(tmaps.size());
    int img_res = input_densities[0].rows();
    std::cout << "Generate inverse maps...\n";

    std::vector<double> density_means(tmaps.size());
    for (int k = 0; k < tmaps.size(); ++k)
    {
        inv_maps[k] = tmaps[k].origin_mesh();
        apply_inverse_map(tmaps[k], inv_maps[k].points(), opts.verbose_level);
        density_means[k] = input_densities[k].mean();

        if (opts.export_maps)
        {
            tmaps[k].fwd_mesh().write(std::string(opts.out_prefix).append("_fwd_").append(make_padded_string(k)).append(".off"));
            inv_maps[k].write(std::string(opts.out_prefix).append("_inv_").append(make_padded_string(k)).append(".off"));
        }
    }

    std::cout << "compute transport field ...\n";
    std::vector<Eigen::Vector2d> vfield;
    compute_transport_field(inv_maps[1], inv_maps[0], vfield);
    save_vectorfield_as_image(std::string(opts.out_prefix).append(make_padded_string(0)).append(".exr").c_str(), vfield, img_res + 1);
}

void compute_transport_field(const Surface_mesh &start, const Surface_mesh &end, std::vector<Eigen::Vector2d> &vfield)
{
    int nv = start.vertices_size();

    vfield.clear();
    vfield.resize(nv);

    for (int j = 0; j < nv; ++j)
    {
        Surface_mesh::Vertex v(j);
        vfield[j] = (end.position(v) - start.position(v) + Eigen::Vector2d::Ones()) * 0.5;
    }
}