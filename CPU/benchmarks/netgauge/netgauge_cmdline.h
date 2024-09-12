/** @file netgauge_cmdline.h
 *  @brief The header file for the command line option parser
 *  generated by GNU Gengetopt version 2.22.1
 *  http://www.gnu.org/software/gengetopt.
 *  DO NOT modify this file, since it can be overwritten
 *  @author GNU Gengetopt by Lorenzo Bettini */

#ifndef NETGAUGE_CMDLINE_H
#define NETGAUGE_CMDLINE_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef NETGAUGE_PARSER_PACKAGE
/** @brief the program name */
#define NETGAUGE_PARSER_PACKAGE "netgauge"
#endif

#ifndef NETGAUGE_PARSER_VERSION
/** @brief the program version */
#define NETGAUGE_PARSER_VERSION "1.0"
#endif

/** @brief Where the command line options are stored */
struct netgauge_cmd_struct
{
  const char *version_help; /**< @brief Print version and exit help description.  */
  int help_flag;	/**< @brief print help (default=off).  */
  const char *help_help; /**< @brief print help help description.  */
  int verbosity_arg;	/**< @brief verbosity level (default='1').  */
  char * verbosity_orig;	/**< @brief verbosity level original value given at command line.  */
  const char *verbosity_help; /**< @brief verbosity level help description.  */
  int server_flag;	/**< @brief act as server (default=off).  */
  const char *server_help; /**< @brief act as server help description.  */
  char * output_arg;	/**< @brief output file name (default='ng.out').  */
  char * output_orig;	/**< @brief output file name original value given at command line.  */
  const char *output_help; /**< @brief output file name help description.  */
  char * full_output_arg;	/**< @brief output file name for all measurements (default='ng-full.out').  */
  char * full_output_orig;	/**< @brief output file name for all measurements original value given at command line.  */
  const char *full_output_help; /**< @brief output file name for all measurements help description.  */
  int tests_arg;	/**< @brief testcount (default='100').  */
  char * tests_orig;	/**< @brief testcount original value given at command line.  */
  const char *tests_help; /**< @brief testcount help description.  */
  int hostnames_flag;	/**< @brief print hostnames (default=off).  */
  const char *hostnames_help; /**< @brief print hostnames help description.  */
  int time_arg;	/**< @brief max time per test (s) (default='100').  */
  char * time_orig;	/**< @brief max time per test (s) original value given at command line.  */
  const char *time_help; /**< @brief max time per test (s) help description.  */
  char * size_arg;	/**< @brief datasize (bytes, from-to) (default='1-131072').  */
  char * size_orig;	/**< @brief datasize (bytes, from-to) original value given at command line.  */
  const char *size_help; /**< @brief datasize (bytes, from-to) help description.  */
  int concurrent_arg;	/**< @brief concurrent (msgs) (default='1').  */
  char * concurrent_orig;	/**< @brief concurrent (msgs)  original value given at command line.  */
  const char *concurrent_help; /**< @brief concurrent (msgs) help description.  */
  char * mode_arg;	/**< @brief transmission mode (default='mpi').  */
  char * mode_orig;	/**< @brief transmission mode original value given at command line.  */
  const char *mode_help; /**< @brief transmission mode help description.  */
  char * comm_pattern_arg;	/**< @brief communication pattern (default='one_one').  */
  char * comm_pattern_orig;	/**< @brief communication pattern original value given at command line.  */
  const char *comm_pattern_help; /**< @brief communication pattern help description.  */
  int grad_arg;	/**< @brief grade of geometrical size distanced (default='2').  */
  char * grad_orig;	/**< @brief grade of geometrical size distanced original value given at command line.  */
  const char *grad_help; /**< @brief grade of geometrical size distanced help description.  */
  int manpage_flag;	/**< @brief write manpage to stdout (default=off).  */
  const char *manpage_help; /**< @brief write manpage to stdout help description.  */
  int init_thread_flag;	/**< @brief initialize with MPI_THREAD_MULTIPLE instead of MPI_THREAD_SINGLE (default=off).  */
  const char *init_thread_help; /**< @brief initialize with MPI_THREAD_MULTIPLE instead of MPI_THREAD_SINGLE help description.  */
  int sanity_check_flag;	/**< @brief perform sanity check of timer (default=off).  */
  const char *sanity_check_help; /**< @brief perform sanity check of timer help description.  */
  
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int verbosity_given ;	/**< @brief Whether verbosity was given.  */
  unsigned int server_given ;	/**< @brief Whether server was given.  */
  unsigned int output_given ;	/**< @brief Whether output was given.  */
  unsigned int full_output_given ;	/**< @brief Whether full-output was given.  */
  unsigned int tests_given ;	/**< @brief Whether tests was given.  */
  unsigned int hostnames_given ;	/**< @brief Whether hostnames was given.  */
  unsigned int time_given ;	/**< @brief Whether time was given.  */
  unsigned int size_given ;	/**< @brief Whether size was given.  */
  unsigned int concurrent_given ;	/**< @brief Whether concurrent was given.  */
  unsigned int mode_given ;	/**< @brief Whether mode was given.  */
  unsigned int comm_pattern_given ;	/**< @brief Whether comm_pattern was given.  */
  unsigned int grad_given ;	/**< @brief Whether grad was given.  */
  unsigned int manpage_given ;	/**< @brief Whether manpage was given.  */
  unsigned int init_thread_given ;	/**< @brief Whether init-thread was given.  */
  unsigned int sanity_check_given ;	/**< @brief Whether sanity-check was given.  */

} ;

/** @brief The additional parameters to pass to parser functions */
struct netgauge_parser_params
{
  int override; /**< @brief whether to override possibly already present options (default 0) */
  int initialize; /**< @brief whether to initialize the option structure netgauge_cmd_struct (default 1) */
  int check_required; /**< @brief whether to check that all required options were provided (default 1) */
  int check_ambiguity; /**< @brief whether to check for options already specified in the option structure netgauge_cmd_struct (default 0) */
  int print_errors; /**< @brief whether getopt_long should print an error message for a bad option (default 1) */
} ;

/** @brief the purpose string of the program */
extern const char *netgauge_cmd_struct_purpose;
/** @brief the usage string of the program */
extern const char *netgauge_cmd_struct_usage;
/** @brief all the lines making the help output */
extern const char *netgauge_cmd_struct_help[];

/**
 * The command line parser
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser (int argc, char * const *argv,
  struct netgauge_cmd_struct *args_info);

/**
 * The command line parser (version with additional parameters - deprecated)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use netgauge_parser_ext() instead
 */
int netgauge_parser2 (int argc, char * const *argv,
  struct netgauge_cmd_struct *args_info,
  int override, int initialize, int check_required);

/**
 * The command line parser (version with additional parameters)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser_ext (int argc, char * const *argv,
  struct netgauge_cmd_struct *args_info,
  struct netgauge_parser_params *params);

/**
 * Save the contents of the option struct into an already open FILE stream.
 * @param outfile the stream where to dump options
 * @param args_info the option struct to dump
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser_dump(FILE *outfile,
  struct netgauge_cmd_struct *args_info);

/**
 * Save the contents of the option struct into a (text) file.
 * This file can be read by the config file parser (if generated by gengetopt)
 * @param filename the file where to save
 * @param args_info the option struct to save
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser_file_save(const char *filename,
  struct netgauge_cmd_struct *args_info);

/**
 * Print the help
 */
void netgauge_parser_print_help(void);
/**
 * Print the version
 */
void netgauge_parser_print_version(void);

/**
 * Initializes all the fields a netgauge_parser_params structure 
 * to their default values
 * @param params the structure to initialize
 */
void netgauge_parser_params_init(struct netgauge_parser_params *params);

/**
 * Allocates dynamically a netgauge_parser_params structure and initializes
 * all its fields to their default values
 * @return the created and initialized netgauge_parser_params structure
 */
struct netgauge_parser_params *netgauge_parser_params_create(void);

/**
 * Initializes the passed netgauge_cmd_struct structure's fields
 * (also set default values for options that have a default)
 * @param args_info the structure to initialize
 */
void netgauge_parser_init (struct netgauge_cmd_struct *args_info);
/**
 * Deallocates the string fields of the netgauge_cmd_struct structure
 * (but does not deallocate the structure itself)
 * @param args_info the structure to deallocate
 */
void netgauge_parser_free (struct netgauge_cmd_struct *args_info);

/**
 * The string parser (interprets the passed string as a command line)
 * @param cmdline the command line stirng
 * @param args_info the structure where option information will be stored
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser_string (const char *cmdline, struct netgauge_cmd_struct *args_info,
  const char *prog_name);
/**
 * The string parser (version with additional parameters - deprecated)
 * @param cmdline the command line stirng
 * @param args_info the structure where option information will be stored
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use netgauge_parser_string_ext() instead
 */
int netgauge_parser_string2 (const char *cmdline, struct netgauge_cmd_struct *args_info,
  const char *prog_name,
  int override, int initialize, int check_required);
/**
 * The string parser (version with additional parameters)
 * @param cmdline the command line stirng
 * @param args_info the structure where option information will be stored
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int netgauge_parser_string_ext (const char *cmdline, struct netgauge_cmd_struct *args_info,
  const char *prog_name,
  struct netgauge_parser_params *params);

/**
 * Checks that all the required options were specified
 * @param args_info the structure to check
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return
 */
int netgauge_parser_required (struct netgauge_cmd_struct *args_info,
  const char *prog_name);


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* NETGAUGE_CMDLINE_H */
