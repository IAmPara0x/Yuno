query = '''
       query ($id: Int) {
         Media (idMal: $id, type: ANIME) {
           id
           idMal
           title {
             romaji
             english
           }
           description
           coverImage{
             large
           }
           synonyms
           siteUrl
           tags{
             id
             name
             description
             category
             rank
           }
           characters(role: MAIN) {
            nodes{
              name {
               first
               middle
               last
               full
               alternative
              }
            }
           }
         }
       }
         '''
