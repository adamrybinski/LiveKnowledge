:- use_module(library(http/http_client)).
:- use_module(library(http/http_json)).
:- use_module(library(readutil)).


% Load environment variables from .env file
load_env :-
    open('.env', read, Stream),
    read_env(Stream),
    close(Stream).


read_env(Stream) :-
    read_line_to_string(Stream, Line),
    (   Line == end_of_file
    ->  true
    ;   (   Line = ""
        ;   sub_string(Line, 0, 1, _, "#")
        )
    ->  read_env(Stream)
    ;   split_string(Line, "=", " ", [Key, Value]),
        setenv(Key, Value),
        read_env(Stream)
    ).


% Get environment variable
get_env(Var, Value) :-
    getenv(Var, Value).


% Call LLM API
call_llm(Prompt, Response) :-
    get_env('LLM_API_KEY', ApiKey),
    get_env('LLM_MODEL', Model),
    get_env('LLM_BASE_URL', BaseUrl),
    atomic_list_concat([BaseUrl, '/chat/completions'], Url),
    atom_concat('Bearer ', ApiKey, AuthHeader),
    http_post(
        Url,
        json(_{
            model: Model,
            messages: [_{role: "user", content: Prompt}]
        }),
        Response,
        [
            request_header('Authorization'=AuthHeader),
            json_object(dict)
        ]
    ).


% Example usage
hello_world :-
    load_env,
    call_llm("Hello, world!", Response),
    writeln(Response).