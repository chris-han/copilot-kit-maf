import { Pencil as LucidePencil, LucideProps } from 'lucide-react';

const Pencil = ({ className, ...props }: LucideProps) => {
  return <LucidePencil className={className} {...props} />;
};

export default Pencil;